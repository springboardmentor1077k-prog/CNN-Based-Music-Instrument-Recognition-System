import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
AUDIO_PATH = './Cleaned_Audio_Dataset' # Use your clean audio
IMG_HEIGHT = 128
IMG_WIDTH = 128
SR = 16000
BATCH_SIZE = 32
EPOCHS = 20

# --- 1. THE MATH FUNCTION (Shared Source of Truth) ---
def audio_to_spectrogram(file_path):
    """
    Converts audio directly to a normalized numpy array.
    No plotting, no images. Just pure math.
    """
    # Load Audio
    y, sr = librosa.load(file_path, sr=SR)
    
    # Generate Mel-Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=IMG_HEIGHT)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Normalize Math (-80dB to 0dB -> 0.0 to 1.0)
    spectrogram_norm = (spectrogram_db + 80) / 80
    spectrogram_norm = np.clip(spectrogram_norm, 0, 1)
    
    # Padding/Cropping to ensure 128 width
    if spectrogram_norm.shape[1] < IMG_WIDTH:
        pad_width = IMG_WIDTH - spectrogram_norm.shape[1]
        spectrogram_norm = np.pad(spectrogram_norm, ((0,0), (0, pad_width)))
    else:
        spectrogram_norm = spectrogram_norm[:, :IMG_WIDTH]
        
    return spectrogram_norm

# --- 2. DATA GENERATOR ---
class RobustGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32, n_classes=11):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.indexes = np.arange(len(self.file_paths))

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, ID in enumerate(indexes):
            file_path = self.file_paths[ID]
            label = self.labels[ID]
            
            # Use the math function
            spec = audio_to_spectrogram(file_path)
            
            X[i,] = np.expand_dims(spec, axis=-1)
            y[i,] = label
            
        return X, y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# --- 3. TRAINING PIPELINE ---
def train_robust_model():
    print("--- 1. Loading File Paths ---")
    file_paths = []
    labels = []
    
    # Sort classes alphabetically to ensure consistency
    classes = sorted([d for d in os.listdir(AUDIO_PATH) if os.path.isdir(os.path.join(AUDIO_PATH, d))])
    num_classes = len(classes)
    class_indices = {cls: i for i, cls in enumerate(classes)}
    
    print(f"✅ Classes found: {classes}")
    
    for cls in classes:
        cls_path = os.path.join(AUDIO_PATH, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith('.wav')]
        for f in files:
            file_paths.append(os.path.join(cls_path, f))
            label = np.zeros(num_classes)
            label[class_indices[cls]] = 1
            labels.append(label)
            
    # Split
    X_train, X_val, y_train, y_val = train_test_split(file_paths, labels, test_size=0.2, random_state=42)
    
    train_gen = RobustGenerator(X_train, y_train, batch_size=BATCH_SIZE, n_classes=num_classes)
    val_gen = RobustGenerator(X_val, y_val, batch_size=BATCH_SIZE, n_classes=num_classes)

    print("\n--- 2. Building Model ---")
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        
        # Note: No 'Rescaling' layer needed because our math function already did 0-1 normalization
        
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Added dropout to prevent overfitting
        layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint = callbacks.ModelCheckpoint('instrunet_robust_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)

    print("\n--- 3. Starting Training ---")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint])
    print("\n✅ DONE: Saved as 'instrunet_robust_model.keras'")

if __name__ == "__main__":
    train_robust_model()