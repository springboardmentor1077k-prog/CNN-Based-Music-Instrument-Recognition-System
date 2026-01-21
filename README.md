# InstruNet AI: CNN-Based Music Instrument Recognition

**InstruNet AI** is a deep learning system designed to automatically identify musical instruments in audio tracks using Convolutional Neural Networks (CNNs). It processes audio into Mel-spectrograms and analyzes them to provide real-time instrument detection.

## 🚀 Live Demo
Experience the model in action:
- **Main Space:** [huggingface.co/spaces/jagathkiran/instrunet-ai](https://huggingface.co/spaces/jagathkiran/instrunet-ai)
- **Direct Interface:** [jagathkiran-instrunet-ai.hf.space](https://jagathkiran-instrunet-ai.hf.space)

---

## ✨ Key Features
- **Real-Time Analysis:** Upload WAV/MP3 files for instant instrument identification.
- **Advanced Slicing:** Automatically divides long audio into 3-second windows for detailed analysis.
- **Visual Feedback:** Side-by-side Waveform and Mel-spectrogram visualizations.
- **Customizable Inference:** Adjust detection thresholds, model sensitivity, and aggregation strategies (Mean/Max).
- **Guest Access:** Try the app instantly with a 10-upload guest limit.
- **Professional Reports:** Export results as JSON or PDF (requires account).

---

## 🛠️ Technology Stack
- **Deep Learning:** TensorFlow / Keras (CNN)
- **Audio Processing:** Librosa, Soundfile
- **Dashboard:** Streamlit
- **Deployment:** Docker / Hugging Face Spaces
- **Reporting:** FPDF, Pandas

---

## 💻 Local Setup & Usage

### 1. Run via Docker (Recommended)
The easiest way to run the dashboard locally with all dependencies pre-configured:

```bash
# Build the production image
docker build -t instrunet-web -f Dockerfile.web .

# Run the container
docker run -p 8501:8501 instrunet-web
```
Open `http://localhost:8501` in your browser.

### 2. Manual Installation
```bash
# Install dependencies
pip install -r requirements-prod.txt

# Run the dashboard
streamlit run src/frontend.py
```

---

## 🏋️ Training & Preprocessing
If you wish to retrain the model or process the dataset:

1. **Preprocessing:**
   ```bash
   python3 src/preprocessing.py
   ```
2. **Training:**
   ```bash
   python3 src/training.py
   ```

---

## 📁 Project Structure
- `src/`: Core logic for audio processing, modeling, and the Streamlit frontend.
- `outputs/`: Contains the trained `.keras` model and benchmark metrics (Confusion Matrix, ROC Curves).
- `docs/`: Technical reports and implementation details.
- `Dockerfile.web`: Optimized Docker configuration for web deployment.

---

## ⚖️ License
This project is for educational and portfolio purposes. Data used is from the IRMAS and NSynth datasets.