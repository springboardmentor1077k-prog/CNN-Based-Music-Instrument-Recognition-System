import os
import cv2
import numpy as np
import tensorflow as tf

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/deep_cnn_best.h5"

# CHANGE THIS IMAGE PATH IF NEEDED
SAMPLE_MEL = r"C:\Users\ADMIN\Downloads\check\pipeline\mel\train\100097.png"

IMG_SIZE = 128
CLASS_NAMES = ["augmented", "clean"]

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_mel(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=-1)   # (128,128,1)
    img = np.expand_dims(img, axis=0)    # (1,128,128,1)

    preds = model.predict(img, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = float(preds[class_id])

    return CLASS_NAMES[class_id], confidence

# =========================
# RUN INFERENCE
# =========================
if __name__ == "__main__":
    label, conf = predict_mel(SAMPLE_MEL)
    print(f"Prediction : {label}")
    print(f"Confidence : {conf:.4f}")
