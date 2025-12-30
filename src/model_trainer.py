import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class ModelTrainer:
    def __init__(self, train_dir, val_dir, img_height=128, img_width=128, batch_size=32):
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
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def build_model(self, dropout_rate=0.5, l2_rate=0.001):
        num_classes = len(self.class_names)

        self.model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
            
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Dropout(dropout_rate),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_rate)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(num_classes) # No activation here, will use from_logits=True in loss
        ])

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        self.model.summary()

    def train(self, epochs=10):
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

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[tensorboard_callback, lr_scheduler, early_stopping]
        )
        return self.history

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
        
        # Get true labels and predicted labels
        y_true = []
        y_pred = []

        # Iterate over the validation dataset
        # Note: unbatch() is important to get individual images
        for images, labels in self.val_ds.unbatch():
            y_true.append(labels.numpy())
            # Predict
            predictions = self.model.predict(tf.expand_dims(images, 0), verbose=0)
            y_pred.append(np.argmax(predictions))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

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
