# CNN-Based-Music-Instrument-Recognition-System
# 🎵 InstruNet AI

**InstruNet AI** is a Streamlit-based web application for **automatic music instrument recognition** using a **CNN-based multilabel deep learning model**. The system detects multiple musical instruments from an audio input and presents results through an interactive dashboard.

This project was developed as part of an **internship / academic project**, covering the complete machine learning lifecycle — preprocessing, model training, evaluation, inference pipeline design, and frontend integration.

---

## ✨ Features

* 🎧 Upload audio files (`.wav`, `.mp3`)
* 🎯 Multilabel music instrument detection
* 📊 Instrument-wise confidence scores
* 🕒 Temporal confidence timelines
* 🎼 Mel-spectrogram visualization
* 📤 Export results as **JSON** and **PDF**
* 🖥️ Streamlit-based interactive UI

---

## 🧠 Model Overview

* **Model Type:** Convolutional Neural Network (CNN)
* **Problem Type:** Multilabel classification
* **Input Representation:** Mel-spectrograms extracted from audio segments
* **Output:** Per-instrument probabilities and confidence scores
* **Techniques Used:**

  * Batch Normalization
  * Regularization
  * Threshold-based decision logic

---

## 📁 Project Structure

The project root directory is **`cnn/`**, which contains both the application code and all experimental notebooks.

```
cnn/
│
├── backend/                    # Inference pipeline, preprocessing & export logic
│   ├── pipeline.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── export.py
│   └── utils.py
│
├── frontend/                   # Streamlit frontend
│   ├── app.py
│   └── requirements.txt
│
├── data/                       # Dataset metadata
│   └── multilabel_labels.csv
│
├── JSON outputs/               # Inference & threshold configuration outputs
│   ├── per_class_thresholds.json
│   └── *.json
│
├── model/                      # Trained CNN model files
│   └── multilabel_cnn_improved.keras
│
├── *.ipynb                     # Jupyter notebooks (preprocessing, training, evaluation)
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** All Jupyter notebooks are intentionally placed directly inside the `cnn/` directory (not grouped into subfolders) to simplify experimentation, comparison, and academic review.

---
📦 Model Files Note

The trained model file (.keras) is intentionally excluded using .gitignore due to file size and deployment constraints.

The repository contains the model architecture, training notebooks, and inference pipeline

The actual trained weights are loaded locally or provided separately during deployment

This approach keeps the repository lightweight and GitHub-friendly

If required, the model file can be shared privately or added later using Git LFS or cloud storage.

## 🌐 Deployment

The application is designed to be deployed using **Streamlit Cloud**.

Deployment characteristics:

* CPU-compatible TensorFlow setup
* Single Streamlit entry point (`frontend/app.py`)
* Explicit dependency management via `requirements.txt`

The deployed app can be **paused, redeployed, or permanently deleted** at any time from the Streamlit Cloud dashboard.

---

## 🔐 Authentication Note

The login functionality implemented in the app is **UI-level only** and is intended purely for demonstration purposes. It does not include backend authentication or user management.

---

## 📦 Dependencies

Major libraries used in this project include:

* `streamlit`
* `tensorflow`
* `librosa`
* `numpy`
* `matplotlib`
* `reportlab`

Refer to `requirements.txt` for the complete dependency list.

---

## 📌 Future Enhancements

* Proper backend authentication
* Improved inference speed and model optimization
* Support for longer and streaming audio inputs
* Advanced analytics and visualization modules

---

## 👩‍💻 Author

**Nandhana M J**
Internship / Academic Project

---

## 📄 License

This project currently does not include a license. A license may be added later if required.
