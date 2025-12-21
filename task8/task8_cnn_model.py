import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Flatten, Dropout, BatchNormalization
)

# Input shape example: (128, 130, 1)
# 128 Mel bands, fixed time steps, 1 channel

model = Sequential()

# -------- Convolution Block 1 --------
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(128, 130, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

# -------- Convolution Block 2 --------
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

# -------- Convolution Block 3 --------
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

# -------- Fully Connected Layers --------
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# -------- Output Layer --------
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
