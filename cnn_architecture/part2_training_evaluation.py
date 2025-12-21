import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay

# import everything from Part 1
from part1_model_definition import (
    model,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    label_encoder
)

# ================= OUTPUT STRUCTURE =================
BASE_OUT = "outputs"
MODEL_DIR = os.path.join(BASE_OUT, "models")
LOG_DIR   = os.path.join(BASE_OUT, "logs")
PLOT_DIR  = os.path.join(BASE_OUT, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
# ===================================================


# ---------- COMPILE MODEL ----------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


# ---------- CALLBACKS ----------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]


# ---------- TRAIN ----------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


# ---------- SAVE TRAINING LOG ----------
pd.DataFrame(history.history).to_csv(
    os.path.join(LOG_DIR, "training_log.csv"),
    index=False
)


# ---------- PLOT LOSS ----------
plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "loss_curve.png"))
plt.close()


# ---------- PLOT ACCURACY ----------
plt.figure()
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "accuracy_curve.png"))
plt.close()


# ---------- TEST EVALUATION ----------
test_probs = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(test_probs, axis=1)

test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Results")
print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {test_prec:.4f}")
print(f"Recall    : {test_rec:.4f}")


# ---------- F1 SCORE ----------
f1 = f1_score(y_true, y_pred, average="weighted")
print(f"F1 Score  : {f1:.4f}")

plt.figure()
plt.bar(["F1 Score"], [f1])
plt.ylim(0, 1)
plt.title("F1 Score")
plt.savefig(os.path.join(PLOT_DIR, "f1_score.png"))
plt.close()


# ---------- CONFUSION MATRIX ----------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
plt.close()
