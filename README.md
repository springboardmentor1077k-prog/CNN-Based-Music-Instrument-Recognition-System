# CNN-Based-Music-Instrument-Recognition-System
# 🎵 InstruNet

InstruNet AI is a deep learning–based music instrument recognition system that detects multiple instruments simultaneously from audio recordings. The system leverages mel-spectrogram representations, a custom Convolutional Neural Network (CNN), and a Streamlit web application for interactive inference and visualization.

## ✨ Features

### 🎼 Core Capabilities
- **Multi-Instrument Detection**: Recognizes 11 different instrument classes
- **Temporal Analysis**: Track instrument presence over time with sliding window approach
- **Polyphonic Support**: Handles multiple simultaneous instruments
- **Confidence Scoring**: Provides probability-based confidence for each detection

### 🖥️ User Interface
- **Secure Authentication**: Password-protected access with session management
- **Interactive Dashboard**: Modern, intuitive Streamlit-based interface
- **Real-time Visualizations**: Waveforms, spectrograms, and intensity timelines
- **Expandable Views**: Detailed probability analysis for all instrument classes

### 📊 Analysis Features
- **Two Aggregation Methods**: Mean and Max
- **Adjustable Thresholds**: Customize detection sensitivity
- **Temporal Smoothing**: Reduce noise with moving average filters
- **Visual Indicators**: Color-coded confidence

### 📤 Export Options
- **JSON Export**: Machine-readable format with complete timeline data
- **PDF Reports**: Professional reports with visualizations and analysis

## 🎸 Supported Instruments

| Code | Instrument | Icon | Code | Instrument | Icon |
|------|-----------|------|------|-----------|------|
| cel  | Cello | 🎻 | org | Organ | 🎹 |
| cla  | Clarinet | 🎷 | pia | Piano | 🎹 |
| flu  | Flute | 🪈 | sax | Saxophone | 🎷 |
| gac  | Acoustic Guitar | 🎸 | tru | Trumpet | 🎺 |
| gel  | Electric Guitar | 🎸 | vio | Violin | 🎻 |
| | | | voi | Voice | 🎤 |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the repository**
   ```bash
   cd Instrunet
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv instrunet_env

   # Windows
   instrunet_env\Scripts\activate

   # Linux/Mac
   source instrunet_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file exists**
   - Ensure `Model/instrunet_irmas_sgd.keras` is present
   - This is the trained CNN model file

5. **Set password (optional)**
   ```bash
   # Windows
   set INSTRUNET_PASSWORD=your_secure_password

   # Linux/Mac
   export INSTRUNET_PASSWORD=your_secure_password
   ```

## 🎯 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Dashboard

1. **Register**
   - Enter your name
   -  Enter your custom password

2. **Login**
   - Enter your name
   -  Enter your custom password

3. **Upload Audio**
   - Click "Choose audio file" in the sidebar
   - Select a WAV or MP3 file
   - File will be displayed with audio player

4. **Configure Analysis**
   - **Aggregation Method**: How to combine segment predictions
     - `mean`: Average confidence across segments
     - `max`: Peak confidence value
   - **Detection Threshold**: Minimum confidence (0.0 - 1.0)
   - **Smoothing Window**: Temporal filter size (1 - 7)

5. **Analyze**
   - Click "Analyze Track"
   - Wait for processing to complete
   - View results in tabs

6. **Explore Results**
   - **Results Tab**: Detected instruments with confidence scores
   - **Visualizations Tab**: Waveforms, spectrograms, timelines
   - **Export Tab**: Download JSON or PDF reports

## 📁 Project Structure

```
Instrunet/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and constants
├── pipeline.py                 # Inference pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── Model/
│   └── instrunet_irmas_sgd.keras  # Trained CNN model
│
├── Utils/
│   ├── __init__.py         
│   ├── aggregation.py         # Prediction aggregation methods
│   ├── audio.py               # Audio loading and preprocessing
│   ├── features.py            # Feature extraction (mel spectrograms)
│   ├── io.py                  # JSON export utilities
│   ├── pdf_report.py          # PDF report generation
│   ├── segmentation.py        # Sliding window segmentation
│   └── visualization.py       # Plotting utilities
│
├── Data/                      # Dataset (not included in repo)
├── Jupyter notebooks/                 # Development notebooks
```

## 🔬 Technical Details

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input**: 128-band log-mel spectrograms
- **Regularization**: Batch normalization + Dropout
- **Training**: SGD optimizer on IRMAS dataset
- **Classes**: 11 instrument categories (multi-label)

### Audio Processing Pipeline
1. **Load Audio**: Convert to mono, resample to 16kHz
2. **Segmentation**: 3-second windows with 1.5-second hop
3. **Preprocessing**: Peak normalization and silence trimming
4. **Feature Extraction**: Log-mel spectrogram generation
5. **Prediction**: CNN inference per segment
6. **Post-processing**: Smoothing and aggregation
7. **Export**: JSON and PDF report generation

### Performance Metrics
- **Micro F1-Score**: 0.6201
- **Macro F1-Score**: 0.609
- **ROC-AUC**: 0.94+
- **Dataset**: IRMAS (6,705 samples)

## 🎓 Academic Context

This project was developed as **Internship Tasks** - the final deliverable of a comprehensive internship focusing on:

- Data preprocessing and augmentation
- CNN architecture design and optimization
- Hyperparameter tuning (kernel size, optimizers, regularization)
- Evaluation metrics and model selection
- Production-ready deployment

### Key Experiments Conducted
- **Task**: Hyperparameter tuning (kernel size: 3×3 → 5×5)
- **Task**: Optimizer comparison (SGD, Adam, RMSProp)
- **Task**: Batch normalization + Dropout regularization
- **Task**: Data segmentation strategy
- **Task**: Aggregation methods evaluation
- **Task**: Intensity timeline generation

## 🔒 Security Notes

### Authentication
- Set custom password via environment variable
- Session-based authentication
- Automatic logout functionality

### Production Deployment
For production use:
```bash
# Set secure password
export INSTRUNET_PASSWORD="your_very_secure_password_here"

# Run with production settings
streamlit run app.py --server.port=8501 --server.headless=true
```

## 📊 Example Outputs

### JSON Export Structure
```json
{
  "audio_file": "music.wav",
  "segment_duration_sec": 3.0,
  "hop_duration_sec": 1.5,
  "aggregation": "max",
  "smoothing": 3,
  "threshold": 0.25,
  "classes": ["cel", "cla", "flu", ...],
  "timeline": [
    {
      "time_sec": 0.0,
      "intensity": {
        "flu": 0.089,
        "org": 0.033,
        ...
      }
    },
    ...
  ]
}
```

### PDF Report Contents
- Analysis metadata and parameters
- Summary statistics
- Mel spectrogram visualization

## 🛠️ Troubleshooting

### Common Issues

**Model file not found**
```
Error: Model file not found: Model/instrunet_irmas_sgd.keras
```
- Ensure the model file exists in the `Model/` directory
- Download from the original repository if missing

**Memory errors during processing**
```
tensorflow.python.framework.errors_impl.ResourceExhaustedError
```
- Close other applications to free memory
- Try shorter audio files
- Reduce batch processing if modified

**Authentication issues**
```
Invalid password
```
- Check if environment variable is set correctly
- Clear browser cache and try again

## 🤝 Contributing

This is an academic project completed for internship requirements. Contributions, suggestions, and feedback are welcome for educational purposes.

## 👥 Authors

Developed as part of Infosys Springboard Virtual Internship 6.0

## 🙏 Acknowledgments

- **Dataset**: IRMAS (Instrument Recognition in Musical Audio Signals)
- **Framework**: TensorFlow/Keras, Streamlit
- **Libraries**: Librosa, NumPy, Matplotlib, ReportLab

## 📧 Contact

For questions or feedback regarding this project, please refer to the project documentation.

---
