import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import random
import cv2 
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 🚀 0. GPU SETUP
# ==========================================
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU DETECTED: {gpus[0].name}")
    except RuntimeError as e: print(e)
else:
    print("❌ NO GPU DETECTED.")

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
DATASET_PATH = 'Spectrogram_Dataset'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 50      
LEARNING_RATE = 0.0001
MIXUP_PROB = 0.6 
RANDOM_SEED = 42

# 🔧 MENTOR TASKS
L2_RATE = 0.001  # "Mild L2"

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ==========================================
# 🛠️ 2. NPY GENERATOR (Standard)
# ==========================================
class NpyMixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder_path, batch_size=32, is_training=True):
        super().__init__()
        self.batch_size = batch_size
        self.is_training = is_training
        self.file_paths = []
        self.labels = []
        
        if not os.path.exists(folder_path): raise ValueError(f"Folder not found: {folder_path}")
        self.classes = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        self.n_classes = len(self.classes)
        
        for cls in self.classes:
            cls_folder = os.path.join(folder_path, cls)
            files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.endswith('.npy')]
            for f in files:
                self.file_paths.append(f)
                self.labels.append(self.class_indices[cls])
        
        self.indexes = np.arange(len(self.file_paths))
        if self.is_training: np.random.shuffle(self.indexes)

    def load_npy(self, file_path):
        try:
            spec = np.load(file_path)
            if spec.shape != (IMG_HEIGHT, IMG_WIDTH): spec = cv2.resize(spec, (IMG_WIDTH, IMG_HEIGHT))
            if len(spec.shape) == 2: spec = np.expand_dims(spec, axis=-1)
            return spec
        except: return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1))

    def __len__(self): return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1))
        y_batch = np.zeros((self.batch_size, self.n_classes))
        for i, idx in enumerate(indexes):
            file_a, label_a = self.file_paths[idx], self.labels[idx]
            spec_a = self.load_npy(file_a)
            if self.is_training and random.random() < MIXUP_PROB:
                rand_idx = random.randint(0, len(self.file_paths) - 1)
                file_b, label_b = self.file_paths[rand_idx], self.labels[rand_idx]
                spec_b = self.load_npy(file_b)
                X_batch[i,] = (spec_a + spec_b) / 2.0
                y_batch[i, label_a] = 1.0; y_batch[i, label_b] = 1.0
            else:
                X_batch[i,] = spec_a; y_batch[i, label_a] = 1.0
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.is_training: np.random.shuffle(self.indexes)

# ==========================================
# 🏗️ 3. UPDATED ARCHITECTURE (Boosted + Regularized)
# ==========================================
def build_mentor_model(n_classes):
    print("🏗️  Building BOOSTED (1024) + REGULARIZED Model...")
    
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Helper for the new "Conv -> BN -> ReLU" order
    def conv_block(x, filters, pool=True):
        # 1. Conv (No Bias, L2 Reg)
        x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(L2_RATE))(x)
        # 2. Batch Norm
        x = layers.BatchNormalization()(x)
        # 3. ReLU
        x = layers.Activation('relu')(x)
        if pool:
            x = layers.MaxPooling2D((2, 2))(x)
        return x

    # Block 1
    x = conv_block(inputs, 32)
    
    # Block 2
    x = conv_block(x, 64)
    
    # Block 3 (Boosted: 256)
    x = conv_block(x, 256)
    
    # Block 4 (Boosted: 512 + Dropout)
    x = conv_block(x, 512)
    x = layers.Dropout(0.3)(x)
    
    # Block 5 (Boosted: 1024 + GAP)
    x = conv_block(x, 1024, pool=False) 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    
    # Dense Head
    x = layers.Dense(1024, use_bias=False, kernel_regularizer=regularizers.l2(L2_RATE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(n_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs, name="InstruNet_Boosted_Reg")
    return model

# ==========================================
# 📊 4. PLOTTING FUNCTION
# ==========================================
def plot_history(history):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    auc = history.history['auc']
    val_auc = history.history['val_auc']
    lr = history.history['lr']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 10))
    
    # 1. Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    # 2. Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    # 3. AUC
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, auc, label='Training AUC')
    plt.plot(epochs_range, val_auc, label='Validation AUC')
    plt.legend(loc='lower right')
    plt.title('Training and Validation AUC')
    plt.grid(True)
    
    # 4. Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, lr, label='Learning Rate', color='orange')
    plt.title('Learning Rate Decay')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_plots.png')
    print("✅ Plots saved as 'training_plots.png'")
    plt.show()

# ==========================================
# 🚀 5. MAIN TRAINING
# ==========================================
def train_full_model():
    train_dir = os.path.join(DATASET_PATH, 'train')
    val_dir = os.path.join(DATASET_PATH, 'validation')
    
    train_gen = NpyMixupGenerator(train_dir, batch_size=BATCH_SIZE, is_training=True)
    val_gen = NpyMixupGenerator(val_dir, batch_size=BATCH_SIZE, is_training=False)
    
    if train_gen.n_classes == 0: return

    # Build Model
    model = build_mentor_model(train_gen.n_classes)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', tf.keras.metrics.AUC(multi_label=True, name='auc')])
    
    # Callbacks
    checkpoint = callbacks.ModelCheckpoint('vk_boosted_reg_model.keras', save_best_only=True, monitor='val_auc', mode='max', verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_auc', patience=12, restore_best_weights=True, mode='max', verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4, verbose=1, mode='max')
    
    # NEW: CSV Logger to store data
    csv_logger = callbacks.CSVLogger('training_log.csv', separator=',', append=False)
    
    print("\n🔥 STARTING BOOSTED + REGULARIZED TRAINING...")
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=EPOCHS,
                        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger])
    
    print("\n✅ Training Complete. Saved as 'vk_boosted_reg_model.keras'")
    
    # Plot Graphs
    plot_history(history)

if __name__ == "__main__":
    train_full_model()