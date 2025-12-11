import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# ✅ CORRECTED PATH: Pointing to your actual image folder
DATASET_PATH = './Mel_Spectrogram_Dataset'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20

def train_solo_fast():
    print(f"--- 1. Loading Images from {DATASET_PATH} ---")
    
    # Check if folder exists first
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Folder '{DATASET_PATH}' not found. Check your directory!")
        return

    # Load Training Data (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale' # Reads the heatmap as intensity (1 Channel)
    )

    # Load Validation Data (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"✅ Found {num_classes} Instruments: {class_names}")

    # --- CRITICAL: ONE-HOT ENCODING ---
    # Converts label '2' -> [0, 0, 1, 0...] so the model understands
    def format_labels(image, label):
        return image, tf.one_hot(label, num_classes)

    train_ds = train_ds.map(format_labels)
    val_ds = val_ds.map(format_labels)

    # Optimization (Cache in memory for speed)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("\n--- 2. Building Model ---")
    model = models.Sequential([
        # Input shape has '1' channel because we selected grayscale
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        
        # Preprocessing: Scale pixels 0-255 -> 0-1
        layers.Rescaling(1./255),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        
        # Output Layer (Sigmoid is used to match our future Multi-Label goals)
        layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Save as 'instrunet_solo_model.keras'
    checkpoint = callbacks.ModelCheckpoint('instrunet_solo_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)

    print("\n--- 3. Starting Fast Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    
    print("\n✅ SOLO Model saved as 'instrunet_solo_model.keras'")

if __name__ == "__main__":
    train_solo_fast()