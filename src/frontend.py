import streamlit as st
import os
import json
import hashlib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import tempfile
import base64
from datetime import datetime

# --- Configuration ---
USER_DB_FILE = "users.json"
APP_TITLE = "Instrunet AI Dashboard"
CLASS_NAMES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
INSTRUMENT_MAP = {
    'cel': 'Cello', 'cla': 'Clarinet', 'flu': 'Flute', 'gac': 'Acoustic Guitar',
    'gel': 'Electric Guitar', 'org': 'Organ', 'pia': 'Piano', 'sax': 'Saxophone',
    'tru': 'Trumpet', 'vio': 'Violin', 'voi': 'Voice'
}

# --- Authentication Functions ---

def load_users():
    if not os.path.exists(USER_DB_FILE):
        return {}
    try:
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_users(users):
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, password):
    return stored_hash == hash_password(password)

def signup(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    
    users[username] = {
        "password": hash_password(password)
    }
    save_users(users)
    return True, "Account created successfully! Please login."

def login(username, password):
    users = load_users()
    if username not in users:
        return False, "Invalid username or password."
    
    if verify_password(users[username]['password'], password):
        return True, "Login successful."
    else:
        return False, "Invalid username or password."

# --- Visualization Functions ---

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Waveform (Amplitude over Time)')
    plt.tight_layout()
    return fig

def plot_mel_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, power=1.0)
    M_db = librosa.amplitude_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram (Frequency-Domain)')
    plt.tight_layout()
    return fig

# --- Export Logic ---

def generate_json_report(result_obj):
    return json.dumps(result_obj, indent=4)

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Instrunet AI - Instrument Detection Report', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

def generate_pdf_report(result_obj, plots=None):
    pdf = PDFReport()
    pdf.add_page()
    
    # Metadata
    pdf.chapter_title("1. Project & File Details")
    details = (f"File Name: {result_obj['metadata']['filename']}\n"
               f"Duration: {result_obj['metadata']['duration']:.2f}s\n"
               f"Analysis Date: {result_obj['metadata']['date']}\n"
               f"Model Threshold: {result_obj['threshold']}")
    pdf.chapter_body(details)
    
    # Summary
    pdf.chapter_title("2. Detected Instruments Summary")
    detected = [p['instrument'] for p in result_obj['predictions'] if p['detected']]
    summary_text = f"Instruments identified: {', '.join(detected) if detected else 'None'}"
    pdf.chapter_body(summary_text)
    
    # Narrative
    pdf.chapter_title("3. Analysis Narrative")
    narrative = ("The audio file was processed using the Instrunet AI CNN model. "
                 "Spectral features were extracted to identify unique harmonic patterns "
                 "associated with various musical instruments.")
    pdf.chapter_body(narrative)

    # Predictions Table
    pdf.chapter_title("4. Detailed Confidence Scores")
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 10, "Instrument", 1)
    pdf.cell(60, 10, "Confidence", 1)
    pdf.cell(60, 10, "Status", 1)
    pdf.ln()
    
    pdf.set_font('Arial', '', 10)
    for p in result_obj['predictions']:
        pdf.cell(60, 10, p['instrument'], 1)
        pdf.cell(60, 10, f"{p['confidence']*100:.1f}%", 1)
        pdf.cell(60, 10, "Detected" if p['detected'] else "Not Detected", 1)
        pdf.ln()

    # Add plots if provided (as file paths)
    if plots:
        pdf.add_page()
        pdf.chapter_title("5. Visualizations")
        for plot_path in plots:
            pdf.image(plot_path, x=10, w=180)
            pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')

# --- Mock Inference ---

def run_mock_inference(filename, y, sr, threshold=0.4, sensitivity=1.0, strategy="Mean"):
    # This simulates the "Single Source of Truth" result object.
    # The REAL model is Single-Label (Softmax), so probabilities sum to 1.0.
    
    predictions = []
    num_classes = len(CLASS_NAMES)
    
    # 1. Simulate Raw Logits (mostly low noise)
    raw_probs = np.random.dirichlet(np.ones(num_classes), size=1)[0]
    
    # 2. Pick a "Winner" (Dominant Instrument)
    # Try to guess from filename for a better demo experience, else random
    winner_idx = np.random.randint(0, num_classes)
    for i, code in enumerate(CLASS_NAMES):
        inst_name = INSTRUMENT_MAP.get(code, code)
        # Simple heuristic check
        if code in filename.lower() or inst_name.lower() in filename.lower():
            winner_idx = i
            break
            
    # Boost the winner to simulate a trained model
    # A good model is 80-99% confident
    confidence_boost = 0.8 + (np.random.random() * 0.15) 
    
    # Re-normalize: Winner gets boost, others get crushed
    raw_probs = raw_probs * (1 - confidence_boost) # Scale down noise
    raw_probs[winner_idx] = confidence_boost
    
    # 3. Apply Strategy & Sensitivity
    for i, code in enumerate(CLASS_NAMES):
        prob = raw_probs[i]
        
        # Strategy Effect (Simulation)
        if strategy == "Max":
            # Max pooling tends to be more confident/aggressive
            prob = prob ** 0.8
        elif strategy == "Mean":
            # Mean pooling flattens peaks slightly
            prob = prob
            
        # Sensitivity Effect
        final_conf = prob * sensitivity
        final_conf = min(max(final_conf, 0.0), 1.0) # Clamp
        
        predictions.append({
            "instrument": INSTRUMENT_MAP.get(code, code),
            "confidence": float(final_conf),
            "detected": bool(final_conf >= threshold)
        })
    
    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    result = {
        "metadata": {
            "filename": filename,
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "sample_rate": int(sr),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "predictions": predictions,
        "threshold": threshold,
        "sensitivity": sensitivity,
        "strategy": strategy
    }
    return result

# --- UI Components ---

def login_page():
    st.title(APP_TITLE)
    st.header("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        # Form submit button
        submitted = st.form_submit_button("Login")
        
        if submitted:
            success, msg = login(username, password)
            if success:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
                
    if st.button("Go to Signup"):
        st.session_state['page'] = 'signup'
        st.rerun()

def signup_page():
    st.title(APP_TITLE)
    st.header("Sign Up")
    
    with st.form("signup_form"):
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submitted = st.form_submit_button("Sign Up")
        
        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            elif not username or not password:
                st.error("Please fill in all fields.")
            else:
                success, msg = signup(username, password)
                if success:
                    st.success(msg)
                    st.session_state['page'] = 'login'
                    st.rerun()
                else:
                    st.error(msg)
    
    if st.button("Back to Login"):
        st.session_state['page'] = 'login'
        st.rerun()

def dashboard_page():
    # Sidebar: User & Advanced Settings
    st.sidebar.title(f"👤 {st.session_state['username']}")
    
    st.sidebar.divider()
    st.sidebar.header("⚙️ Advanced Settings")
    
    # 1. Detection Threshold
    threshold = st.sidebar.slider(
        "Detection Threshold", 
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="Minimum confidence score required for an instrument to be marked as 'Detected'."
    )
    
    # 2. Model Sensitivity
    sensitivity = st.sidebar.slider(
        "Model Sensitivity", 
        min_value=0.5, max_value=1.5, value=1.0, step=0.1,
        help="Boost or suppress all confidence scores. High sensitivity helps find faint instruments but increases noise."
    )
    
    # 3. Aggregation Strategy
    strategy = st.sidebar.selectbox(
        "Aggregation Strategy",
        options=["Mean", "Max"],
        index=0,
        help="How to combine scores from different 3-second windows: Mean (Average) or Max (Peak Confidence)."
    )
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['prediction_result'] = None
        st.rerun()
    
    st.title("🎵 Audio Analysis Dashboard")
    
    # 1. Input & Controls Section
    st.header("1. Input & Controls")
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=['wav', 'mp3'])
    
    if uploaded_file is not None:
        # Load audio data into session state to avoid re-loading
        if 'audio_data' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
            y, sr = librosa.load(uploaded_file, sr=None)
            st.session_state['audio_data'] = (y, sr)
            st.session_state['last_file'] = uploaded_file.name
            st.session_state['prediction_result'] = None # Clear old results

        y, sr = st.session_state['audio_data']
        
        # Use vertical_alignment to align the button with the audio player
        col_audio, col_btn = st.columns([3, 1], vertical_alignment="bottom")
        with col_audio:
            st.audio(uploaded_file)
        
        run_clicked = False
        with col_btn:
            if st.button("🚀 Run Prediction", width="stretch"):
                run_clicked = True
        
        # Handle prediction logic outside the columns to avoid layout shifts
        if run_clicked:
            with st.spinner("Analyzing instruments..."):
                # Step 1: Create Result Object (Single Source of Truth)
                res = run_mock_inference(
                    uploaded_file.name, y, sr, 
                    threshold=threshold, 
                    sensitivity=sensitivity, 
                    strategy=strategy
                )
                st.session_state['prediction_result'] = res
                st.success(f"Analysis Complete (Strategy: {strategy})!")

        # 2. Audio Visualization Section
        st.divider()
        
        with st.expander("2. Audio Visualization", expanded=True):
            tab1, tab2 = st.tabs(["Waveform", "Spectrogram"])
            
            with tab1:
                fig_wav = plot_waveform(y, sr)
                st.pyplot(fig_wav)
                # Store plot for PDF
                st.session_state['plot_wav_path'] = os.path.join(tempfile.gettempdir(), "temp_wav.png")
                fig_wav.savefig(st.session_state['plot_wav_path'])

            with tab2:
                fig_spec = plot_mel_spectrogram(y, sr)
                st.pyplot(fig_spec)
                # Store plot for PDF
                st.session_state['plot_spec_path'] = os.path.join(tempfile.gettempdir(), "temp_spec.png")
                fig_spec.savefig(st.session_state['plot_spec_path'])

        # 3. Prediction Results Section
        if st.session_state.get('prediction_result'):
            st.divider()
            st.header("3. Prediction Results")
            res = st.session_state['prediction_result']
            
            # Summary
            detected = [p['instrument'] for p in res['predictions'] if p['detected']]
            st.subheader("Detected Instruments")
            if detected:
                cols = st.columns(len(detected))
                for i, inst in enumerate(detected):
                    cols[i].success(f"✅ {inst}")
            else:
                st.info("No instruments detected above the threshold.")

            # Confidence Visualization
            st.subheader("Confidence Scores")
            for p in res['predictions']:
                color = "green" if p['detected'] else "gray"
                # Accessibility: Label says clearly what it is
                label = f"{p['instrument']} ({p['confidence']*100:.1f}%)"
                st.progress(p['confidence'], text=label)
            
            # Export Buttons (Only appear after prediction)
            st.divider()
            st.subheader("📤 Export Results")
            col_json, col_pdf = st.columns(2)
            
            with col_json:
                json_data = generate_json_report(res)
                st.download_button(
                    label="📄 Download JSON (Facts)",
                    data=json_data,
                    file_name=f"analysis_{uploaded_file.name}.json",
                    mime="application/json"
                )
            
            with col_pdf:
                # Generate PDF using session state data
                pdf_bytes = generate_pdf_report(res, plots=[
                    st.session_state.get('plot_wav_path'),
                    st.session_state.get('plot_spec_path')
                ])
                st.download_button(
                    label="📕 Download PDF Report (Story)",
                    data=pdf_bytes,
                    file_name=f"report_{uploaded_file.name}.pdf",
                    mime="application/pdf"
                )

    # 4. Power User Section (Model Metrics)
    st.divider()
    with st.expander("🔧 Model Performance Metrics (Power Users)"):
        st.write("These metrics represent the overall performance of the model on the test dataset.")
        
        # Paths to static assets
        # Go up two levels: src/frontend.py -> src -> root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(project_root, "outputs")
        conf_matrix_path = os.path.join(outputs_dir, "normalized_confusion_matrix.png")
        roc_path = os.path.join(outputs_dir, "roc_curves.png")
        report_path = os.path.join(outputs_dir, "classification_report.csv")

        # 1. Classification Report (Table)
        if os.path.exists(report_path):
            st.subheader("Global Classification Report")
            try:
                import pandas as pd
                df_metrics = pd.read_csv(report_path)
                st.dataframe(df_metrics, width="stretch")
            except Exception as e:
                st.error(f"Could not load metrics table: {e}")
        
        # 2. Visualizations (Columns)
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("Confusion Matrix")
            if os.path.exists(conf_matrix_path):
                st.image(conf_matrix_path, caption="Normalized Confusion Matrix", width="stretch")
            else:
                st.warning("Confusion Matrix image not found.")

        with col_m2:
            st.subheader("ROC Curves")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Receiver Operating Characteristic (ROC)", width="stretch")
            else:
                st.warning("ROC Curves image not found.")

# --- Main App Logic ---

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎵", layout="wide")
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
        
    if st.session_state['logged_in']:
        dashboard_page()
    else:
        if st.session_state['page'] == 'login':
            login_page()
        else:
            signup_page()

if __name__ == "__main__":
    main()