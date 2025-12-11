import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import librosa
import os
import random
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
AUDIO_PATH = './Cleaned_Audio_Dataset'
SOLO_MODEL_PATH = 'instrunet_solo_model.keras' # Load the smart brain we just made
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 15 # Short training session to adapt to mixing
SR = 16000

# --- 1. DYNAMIC MIX GENERATOR (The Chef) ---
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
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        X = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, ID in enumerate(list_ids_temp):
            # LOGIC: 50% Mix (Learn Polyphony), 50% Solo (Don't forget basics)
            if self.is_training and random.random() > 0.5:
                # --- MIXING ---
                file_a, label_a = self.file_paths[ID], self.labels[ID]
                rand_idx = random.randint(0, len(self.file_paths) - 1)
                file_b, label_b = self.file_paths[rand_idx], self.labels[rand_idx]
                
                wav_a, _ = librosa.load(file_a, sr=SR)
                wav_b, _ = librosa.load(file_b, sr=SR)
                
                min_len = min(len(wav_a), len(wav_b))
                wav_mixed = (wav_a[:min_len] + wav_b[:min_len]) / 2
                label_final = np.maximum(label_a, label_b)
                
            else:
                # --- SOLO ---
                file_path = self.file_paths[ID]
                label_final = self.labels[ID]
                wav_mixed, _ = librosa.load(file_path, sr=SR)

            # Generate Spectrogram
            spectrogram = librosa.feature.melspectrogram(y=wav_mixed, sr=SR, n_mels=IMG_HEIGHT)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            spectrogram_norm = (spectrogram_db + 80) / 80
            spectrogram_norm = np.clip(spectrogram_norm, 0, 1)
            
            if spectrogram_norm.shape[1] < IMG_WIDTH:
                pad = IMG_WIDTH - spectrogram_norm.shape[1]
                spectrogram_norm = np.pad(spectrogram_norm, ((0,0), (0, pad)))
            else:
                spectrogram_norm = spectrogram_norm[:, :IMG_WIDTH]

            X[i,] = np.expand_dims(spectrogram_norm, axis=-1)
            y[i,] = label_final

        return X, y

# --- 2. SETUP DATA ---
def load_data():
    file_paths = []
    labels = []
    classes = sorted([d for d in os.listdir(AUDIO_PATH) if os.path.isdir(os.path.join(AUDIO_PATH, d))])
    num_classes = len(classes)
    class_indices = {cls: i for i, cls in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(AUDIO_PATH, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith('.wav')]
        for f in files:
            file_paths.append(os.path.join(cls_path, f))
            label = np.zeros(num_classes)
            label[class_indices[cls]] = 1
            labels.append(label)
    return np.array(file_paths), np.array(labels), num_classes

# --- 3. FINE-TUNING PROCESS ---
def train_finetune():
    print("--- 1. Loading Pre-Trained Solo Model ---")
    base_model = tf.keras.models.load_model(SOLO_MODEL_PATH)
    
    # We lower the learning rate drastically (from 0.001 to 0.0001)
    # This ensures we don't "break" the knowledge it already has.
    base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    
    print("✅ Model Loaded. Switching to Mixed Data Generator...")
    
    files, labels, num_classes = load_data()
    X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=0.2, random_state=42)
    
    train_gen = DynamicMixDataGenerator(X_train, y_train, batch_size=BATCH_SIZE, n_classes=num_classes, is_training=True)
    val_gen = DynamicMixDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, n_classes=num_classes, is_training=False)
    
    print("\n--- 2. Starting Fine-Tuning (Teaching it to Mix) ---")
    checkpoint = callbacks.ModelCheckpoint('instrunet_poly_model.keras', save_best_only=True, monitor='val_loss', verbose=1)
    
    base_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    
    print("\n✅ POLYPHONIC Model saved as 'instrunet_poly_model.keras'")

if __name__ == "__main__":
    train_finetune()