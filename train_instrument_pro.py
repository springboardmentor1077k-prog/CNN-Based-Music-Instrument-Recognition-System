import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, metrics
import numpy as np
import librosa
import os
import random
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
AUDIO_PATH = './Cleaned_Audio_Dataset' # Must point to your .wav files
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 30 
SR = 16000

# --- 2. THE DYNAMIC DATA GENERATOR (The "Chef") ---
class DynamicMixDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32, n_classes=11, is_training=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.is_training = is_training 
        self.indexes = np.arange(len(self.file_paths))

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        # Shuffle data after every epoch so batches are always different
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        X = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, ID in enumerate(list_ids_temp):
            # LOGIC: 50% Mix (Simulate Polyphony), 50% Solo (Learn Fundamentals)
            if self.is_training and random.random() > 0.5:
                # --- A. MIXING MODE ---
                file_a, label_a = self.file_paths[ID], self.labels[ID]
                
                # Pick random File B
                rand_idx = random.randint(0, len(self.file_paths) - 1)
                file_b, label_b = self.file_paths[rand_idx], self.labels[rand_idx]
                
                # Load & Mix Audio
                wav_a, _ = librosa.load(file_a, sr=SR)
                wav_b, _ = librosa.load(file_b, sr=SR)
                
                # Match lengths safely
                min_len = min(len(wav_a), len(wav_b))
                wav_mixed = (wav_a[:min_len] + wav_b[:min_len]) / 2
                
                # Combine Labels (Union)
                label_final = np.maximum(label_a, label_b)
                
            else:
                # --- B. SOLO MODE ---
                file_path = self.file_paths[ID]
                label_final = self.labels[ID]
                wav_mixed, _ = librosa.load(file_path, sr=SR)

            # --- C. GENERATE SPECTROGRAM ---
            # Generate Mel-Spectrogram
            spectrogram = librosa.feature.melspectrogram(y=wav_mixed, sr=SR, n_mels=IMG_HEIGHT)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            
            # Normalize (Crucial for Neural Networks: Map -80dB...0dB to 0...1)
            # We assume range is approx -80dB to 0dB
            spectrogram_norm = (spectrogram_db + 80) / 80
            spectrogram_norm = np.clip(spectrogram_norm, 0, 1) # Clamp values
            
            # Fix Width to exactly 128 pixels (Padding/Cropping)
            curr_width = spectrogram_norm.shape[1]
            if curr_width < IMG_WIDTH:
                pad_width = IMG_WIDTH - curr_width
                spectrogram_norm = np.pad(spectrogram_norm, ((0,0), (0, pad_width)))
            else:
                spectrogram_norm = spectrogram_norm[:, :IMG_WIDTH]

            # Reshape for CNN (Height, Width, Channel)
            X[i,] = np.expand_dims(spectrogram_norm, axis=-1)
            y[i,] = label_final

        return X, y

# --- 3. DATA LOADING HELPER ---
def load_and_map_data():
    if not os.path.exists(AUDIO_PATH):
        raise FileNotFoundError(f"❌ Error: {AUDIO_PATH} not found. Run preprocessing first!")

    file_paths = []
    labels = []
    
    # Get sorted class names (e.g., ['cel', 'cla', 'gac'...])
    classes = sorted([d for d in os.listdir(AUDIO_PATH) if os.path.isdir(os.path.join(AUDIO_PATH, d))])
    num_classes = len(classes)
    print(f"✅ Found {num_classes} Instrument Classes: {classes}")
    
    # Map 'pia' -> 0, 'gel' -> 1, etc.
    class_indices = {cls: i for i, cls in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(AUDIO_PATH, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith('.wav')]
        
        for f in files:
            file_paths.append(os.path.join(cls_path, f))
            # Create One-Hot Label
            label = np.zeros(num_classes)
            label[class_indices[cls]] = 1
            labels.append(label)
            
    return np.array(file_paths), np.array(labels), num_classes

# --- 4. MAIN TRAINING PIPELINE ---
def train_pro_model():
    print("\n--- 1. Preparing Data ---")
    files, labels, num_classes = load_and_map_data()
    
    # Split Data: 80% Train, 20% Validation
    X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} files | Validating on {len(X_val)} files")
    
    # Create Generators
    train_gen = DynamicMixDataGenerator(X_train, y_train, batch_size=BATCH_SIZE, n_classes=num_classes, is_training=True)
    val_gen = DynamicMixDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, n_classes=num_classes, is_training=False)

    print("\n--- 2. Building Model Architecture ---")
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Output Layer: Sigmoid for Multi-Label
        layers.Dense(num_classes, activation='sigmoid')
    ])

    # Compile with Pro Metrics
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )

    # Smart Callbacks
    callbacks_list = [
        # Save best model
        callbacks.ModelCheckpoint('instrunet_pro_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
        # Stop if not learning
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        # Slow down learning rate if stuck (Fine-tuning)
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    print("\n--- 3. Starting Training ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks_list
    )
    
    print("\n✅ SUCCESS: Model saved as 'instrunet_pro_model.keras'")

if __name__ == "__main__":
    train_pro_model()