import os

# Fix for Numba/Librosa permission issue in Docker (non-root)
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib
matplotlib.use('Agg')  # Required for headless plotting on HPC
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight


class ModelTrainer:
    def __init__(self, train_dir, val_dir, img_height=224, img_width=224, batch_size=32):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.train_ds = None
        self.val_ds = None
        self.class_names = None
        self.model = None
        self.history = None

    def load_data(self):
        print("Loading training dataset...")
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            color_mode='rgb',
            shuffle=True
        )
        
        print("Loading validation dataset...")
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.val_dir,
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            color_mode='rgb',
            shuffle=False
        )

        self.class_names = self.train_ds.class_names
        print(f"Found {len(self.class_names)} classes: {self.class_names}")

        # Configure for performance
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Apply SpecAugment to training data only
        self.train_ds = self.train_ds.map(
            lambda x, y: (self.apply_spec_augment(x), y),
            num_parallel_calls=AUTOTUNE
        )

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def apply_spec_augment(self, image_batch, freq_mask_max=20, time_mask_max=30):
        """Applies random frequency and time masking to a batch of images."""
        # Note: image_batch is (Batch, H, W, C)
        batch_size = tf.shape(image_batch)[0]
        img_height = tf.shape(image_batch)[1]
        img_width = tf.shape(image_batch)[2]

        # 1. Frequency Masking (Horizontal stripes)
        f = tf.random.uniform(shape=[batch_size], minval=0, maxval=freq_mask_max, dtype=tf.int32)
        f0 = tf.random.uniform(shape=[batch_size], minval=0, maxval=img_height - freq_mask_max, dtype=tf.int32)
        
        # 2. Time Masking (Vertical stripes)
        t = tf.random.uniform(shape=[batch_size], minval=0, maxval=time_mask_max, dtype=tf.int32)
        t0 = tf.random.uniform(shape=[batch_size], minval=0, maxval=img_width - time_mask_max, dtype=tf.int32)

        # Create masks
        # We process this via a loop over the batch for simplicity in TF
        def mask_single_image(i):
            img = image_batch[i]
            
            # Apply freq mask
            mask_f = tf.concat([
                tf.ones([f0[i], img_width, 3]),
                tf.zeros([f[i], img_width, 3]),
                tf.ones([img_height - f0[i] - f[i], img_width, 3])
            ], axis=0)
            img = img * mask_f
            
            # Apply time mask
            mask_t = tf.concat([
                tf.ones([img_height, t0[i], 3]),
                tf.zeros([img_height, t[i], 3]),
                tf.ones([img_height, img_width - t0[i] - t[i], 3])
            ], axis=1)
            img = img * mask_t
            
            return img

        # Use tf.map_fn to apply masking to each image in the batch
        augmented_batch = tf.map_fn(mask_single_image, tf.range(batch_size), dtype=tf.float32)
        
        return augmented_batch


    def build_model(self, dropout_rate=0.5, l2_rate=0.001):
        num_classes = len(self.class_names)

        self.model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
            
            # Layer 1: Increased filters and larger kernel to capture broad structures
            layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Layer 2: 64 filters
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Layer 3: 128 filters
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            # Layer 4: 256 filters for deep feature extraction
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Dropout(dropout_rate),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_rate)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(num_classes) # Logits
        ])

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        self.model.summary()

    def train(self, epochs=10, use_class_weights=True):
        if self.model is None:
            print("Model not built. Call build_model() first.")
            return

        print(f"Starting training for {epochs} epochs...")
        
        # Callbacks
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        # Learning Rate Scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=1e-6,
            verbose=1
        )

        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        # Calculate class weights if requested
        weights = None
        if use_class_weights:
            weights = self.get_class_weights()

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[tensorboard_callback, lr_scheduler, early_stopping],
            class_weight=weights
        )
        return self.history

    def get_class_weights(self):
        """Calculates class weights to handle imbalance."""
        if self.train_ds is None:
            print("Dataset not loaded. Call load_data() first.")
            return None
            
        print("Calculating class weights...")
        
        # Extract labels from the dataset
        y_train = []
        # We need to unbatch to get individual labels correctly if we want to be sure,
        # but iterating over batches and extending also works.
        for _, labels in self.train_ds:
            y_train.extend(labels.numpy())
        y_train = np.array(y_train)
        
        # Calculate weights
        unique_classes = np.unique(y_train)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        
        weight_dict = {cls: weight for cls, weight in zip(unique_classes, weights)}
        
        # Print for verification
        readable_weights = {self.class_names[i]: f"{w:.2f}" for i, w in weight_dict.items()}
        print(f"Computed Class Weights: {readable_weights}")
        
        return weight_dict


    def plot_history(self, output_dir):
        if self.history is None:
            print("No training history found.")
            return

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
        plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Training history plot saved to {plot_path}")

    def save_model(self, output_dir):
        model_path = os.path.join(output_dir, 'instrunet_cnn.keras')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def evaluate_model(self, output_dir):
        if self.model is None or self.val_ds is None:
            print("Model or validation data not available.")
            return

        print("Evaluating model on validation set...")
        
        # Get true labels efficiently
        y_true = []
        for _, labels in self.val_ds:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)

        # Get predicted labels using batch prediction
        print("Running batch prediction...")
        predictions = self.model.predict(self.val_ds)
        # Use softmax if the model outputs logits (it does, based on build_model)
        if predictions.shape[-1] > 1:
            y_pred = np.argmax(predictions, axis=1)
        else:
            y_pred = (predictions > 0.5).astype(int)

        # 1. Classification Report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(output_dir, 'classification_report.csv')
        report_df.to_csv(report_path)
        print(f"Classification report saved to {report_path}")

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    # Example usage (Task 7 verification)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-ProcessedTrainingData", "train", "spectrograms")
    VAL_DIR = os.path.join(PROJECT_ROOT, "datasets", "IRMAS-ProcessedTrainingData", "validation", "spectrograms")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    
    # Ensure data exists
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"Data directories not found.\nTrain: {TRAIN_DIR}\nVal: {VAL_DIR}")
        exit(1)

    trainer = ModelTrainer(TRAIN_DIR, VAL_DIR, batch_size=32)
    trainer.load_data()
    trainer.build_model()
    # Train for a few epochs to verify pipeline
    trainer.train(epochs=3)
    trainer.plot_history(OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
