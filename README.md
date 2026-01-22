# 🎵 InstruNetAI: Polyphonic Instrument Recognition System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge)

**InstruNetAI** is an advanced Deep Learning system designed to automatically detect and classify musical instruments in polyphonic audio tracks. Unlike simple classifiers, InstruNetAI can identify **multiple instruments playing simultaneously** (e.g., "Violin + Piano") and map their presence over time.

The project features a production-ready **Streamlit Dashboard** complete with user authentication, spectral analysis, and automated PDF reporting capabilities.

---

## ✨ Key Features

### 🧠 Core Intelligence
* **Polyphonic Detection:** Capable of recognizing 11 distinct instrument classes simultaneously (Cello, Clarinet, Flute, Guitar (Ac/El), Organ, Piano, Saxophone, Trumpet, Violin, Voice).
* **Temporal Analysis:** Uses a sliding-window approach to detect which instruments are active at specific seconds of the track.
* **Robust CNN Model:** Built on a custom Convolutional Neural Network with L2 Regularization and Global Average Pooling to ensure high generalization on the IRMAS dataset.

### 💻 Interactive Dashboard
* **Secure Authentication:** Session-based Login system to protect the dashboard.
* **Real-Time Visualization:** Generates interactive **Mel-Spectrograms** and **Timeline Heatmaps** using Plotly.
* **Smart Filtering:** Adjustable "Detection Sensitivity" slider to filter out background noise or low-confidence predictions.

### 📄 Reporting & Export
* **Automated PDF Reports:** Generates professional-grade PDF summaries including file metadata, detected instruments, and spectral images.
* **Email Integration:** Sends the full analysis report (PDF + JSON) directly to the user's email via SMTP.
* **Data Export:** Download raw analysis data in JSON format for further research.

---

## 🛠️ Technical Architecture

### Model Design
The heart of InstruNetAI is a **Regularized CNN** trained on 128x128 Log-Mel Spectrograms.
* **Input:** Audio chunks (3.0s duration) converted to Mel-Spectrograms.
* **Backbone:** 5 Convolutional Blocks (increasing filters up to 1024).
* **Stabilization:** Batch Normalization applied *before* ReLU activation.
* **Regularization:** L2 Weight Decay + Progressive Dropout (0.3 $\to$ 0.5) to prevent overfitting.

### Performance Metrics (IRMAS Test Set)
| Metric | Score | Significance |
| :--- | :--- | :--- |
| **Top-2 Accuracy** | **86.01%** | The correct instrument is within the top 2 predictions 86% of the time. |
| **Samples F1** | **0.541** | Balanced precision/recall for multi-label classification. |
| **Inference Speed** | **~0.2s** | Real-time processing per 3-second chunk on CPU. |

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8 or higher
* Git

### 1. Clone the Repository
```bash
# Clone the repository to your local machine
git clone [https://github.com/Vaibhavakanna-P/InstruNetAI.git](https://github.com/Vaibhavakanna-P/InstruNetAI.git)
cd InstruNetAI
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure Secrets (For Email Feature)
To use the email sending feature, create a .streamlit/secrets.toml file (locally) or configure it in Streamlit Cloud:

```bash
email_username = "your-email@gmail.com"
email_password = "your-16-char-app-password"
```
### 4. Run the Application
```bash
streamlit run vk_dashboard.py
The app will open in your browser at http://localhost:8501.

📂 Project Structure
Plaintext
InstruNetAI/
├── vk_dashboard.py             # Main Streamlit Application (Frontend & Logic)
├── vk_boosted_reg_model.keras  # Trained TensorFlow Model
├── requirements.txt            # Project Dependencies
├── packages.txt                # System-level dependencies (libsndfile1)
├── assets/                     # Images (Logo, Background)
│   ├── background.jpeg
│   └── logo.jpeg
└── README.md                   # Project Documentation

```
### 📸 Usage Guide
Login: Enter your username and email to access the system.
Upload: Drag and drop an audio file (.wav or .mp3).
Analyze: The system automatically processes the audio:
Visuals: View the Mel-Spectrogram.
Timeline: See exactly when each instrument enters/leaves.

### Export:
Click "Generate PDF Report" to create a summary.
Click "Share via Email" to send the report to yourself or a colleague.

### 🤝 Acknowledgments
Dataset: IRMAS (Instrument Recognition in Musical Audio Signals)
Frameworks: TensorFlow, Streamlit, Librosa.
Mentorship: Developed under the guidance of Springboard Mentor
