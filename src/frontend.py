import streamlit as st
import os
import sys
import json
import hashlib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from fpdf import FPDF
import tempfile
import base64
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualizer import save_clean_spectrogram

# Lazy import TensorFlow to avoid startup lag
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Configuration ---
USER_DB_FILE = "users.json"
APP_TITLE = "Instrunet AI"
CLASS_NAMES = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]
INSTRUMENT_MAP = {
    "cel": "Cello",
    "cla": "Clarinet",
    "flu": "Flute",
    "gac": "Acoustic Guitar",
    "gel": "Electric Guitar",
    "org": "Organ",
    "pia": "Piano",
    "sax": "Saxophone",
    "tru": "Trumpet",
    "vio": "Violin",
    "voi": "Voice",
}

# --- Authentication Functions ---


def load_users():
    if not os.path.exists(USER_DB_FILE):
        return {}
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_users(users):
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, password):
    return stored_hash == hash_password(password)

def signup(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists."

    users[username] = {"password": hash_password(password)}
    save_users(users)
    return True, "Account created successfully! Please login."

def login(username, password):
    users = load_users()
    if username not in users:
        return False, "Invalid username or password."

    if verify_password(users[username]["password"], password):
        return True, "Login successful."
    else:
        return False, "Invalid username or password."


# --- Visualization Functions ---


def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    return fig

def plot_mel_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, power=1.0)
    M_db = librosa.amplitude_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    ax.set_ylabel("Frequency (Mel)")
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    return fig

def plot_timeline_linechart(timeline_data, smoothing_window=1):
    # Prepare data for Line Chart
    # Convert list of dicts to DataFrame for easier manipulation
    records = []
    for t_idx, window in enumerate(timeline_data):
        for code in CLASS_NAMES:
            records.append(
                {
                    "Time (s)": window["start_time"],
                    "Instrument": INSTRUMENT_MAP.get(code, code),
                    "Confidence": window["scores"][code],
                }
            )

    df = pd.DataFrame(records)

    # Apply Smoothing (Moving Average) if window > 1
    if smoothing_window > 1:
        # Group by Instrument and apply rolling mean
        df["Confidence"] = df.groupby("Instrument")["Confidence"].transform(
            lambda x: x.rolling(window=smoothing_window, min_periods=1).mean()
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Time (s)",
        y="Confidence",
        hue="Instrument",
        ax=ax,
        palette="tab10",
        linewidth=2,
    )

    ax.set_title(f"Instrument Activation Timeline (Smoothing: {smoothing_window})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Confidence Score")
    plt.tight_layout()
    return fig

def plot_timeline_heatmap(timeline_data, smoothing_window=1):
    # Prepare data for Heatmap
    records = []
    for t_idx, window in enumerate(timeline_data):
        for code in CLASS_NAMES:
            records.append(
                {
                    "Time (s)": window["start_time"],
                    "Instrument": INSTRUMENT_MAP.get(code, code),
                    "Confidence": window["scores"][code],
                }
            )

    df = pd.DataFrame(records)

    # Apply Smoothing
    if smoothing_window > 1:
        df["Confidence"] = df.groupby("Instrument")["Confidence"].transform(
            lambda x: x.rolling(window=smoothing_window, min_periods=1).mean()
        )

    # Pivot for Heatmap: Rows=Instruments, Cols=Time
    pivot_df = df.pivot(index="Instrument", columns="Time (s)", values="Confidence")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_df, cmap="Greens", ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Confidence"}
    )
    ax.set_title(f"Instrument Activation Heatmap (Smoothing: {smoothing_window})")
    plt.tight_layout()
    return fig


# --- Export Logic ---


def generate_json_report(result_obj):
    return json.dumps(result_obj, indent=4)


class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "Instrument Detection Report", 0, 1, "C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, body)
        self.ln()


def generate_pdf_report(result_obj, plots=None):
    pdf = PDFReport()
    pdf.add_page()

    # Metadata
    pdf.chapter_title("Project & File Details")
    details = (
        f"File Name: {result_obj['metadata']['filename']}
"
        f"Duration: {result_obj['metadata']['duration']:.2f}s
"
        f"Analysis Date: {result_obj['metadata']['date']}
"
        f"Model Threshold: {result_obj['threshold']}"
    )
    pdf.chapter_body(details)

    # Summary
    pdf.chapter_title("Detected Instruments Summary")
    detected = [p["instrument"] for p in result_obj["predictions"] if p["detected"]]
    summary_text = (
        f"Instruments identified: {', '.join(detected) if detected else 'None'}"
    )
    pdf.chapter_body(summary_text)

    # Narrative
    pdf.chapter_title("Analysis Narrative")
    narrative = (
        "The audio file was processed using the Instrunet AI CNN model. "
        "Spectral features were extracted to identify unique harmonic patterns "
        "associated with various musical instruments."
    )
    pdf.chapter_body(narrative)

    # Predictions Table
    pdf.chapter_title("Detailed Confidence Scores")
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 10, "Instrument", 1)
    pdf.cell(60, 10, "Confidence", 1)
    pdf.cell(60, 10, "Status", 1)
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for p in result_obj["predictions"]:
        pdf.cell(60, 10, p["instrument"], 1)
        pdf.cell(60, 10, f"{p['confidence'] * 100:.1f}%", 1)
        pdf.cell(60, 10, "Detected" if p["detected"] else "Not Detected", 1)
        pdf.ln()

    # Add plots if provided (as file paths)
    if plots:
        pdf.add_page()
        pdf.chapter_title("5. Visualizations")

        titles = [
            "Waveform Analysis (Time Domain)",
            "Spectral Analysis (Frequency Domain)",
            "Temporal Timeline Analysis",
        ]

        for i, plot_path in enumerate(plots):
            # Check for page break before adding new plot
            # Approx height needed: Title (10) + Image (approx 100) + Margin (10) = 120
            if pdf.get_y() > 170: 
                pdf.add_page()

            # If we have a title for this plot index, use it
            if i < len(titles):
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 10, titles[i], 0, 1, "L")

            pdf.image(plot_path, x=10, w=180)
            pdf.ln(5)

    return pdf.output(dest="S").encode("latin-1")


# --- Styling ---


def load_css():
    st.markdown(
        """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
h1 {color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; text-align: center; margin-bottom: 0.5rem;}
h2, h3 {color: #34495e; font-family: 'Helvetica Neue', sans-serif;}
.icon-mar {margin-right: 10px; color: #4CAF50;}
.stExpander {background-color: #f8f9fa; border-radius: 10px; border: 1px solid #e9ecef;}
.stButton button {background-color: #4CAF50; color: white; border-radius: 8px; font-weight: bold;}
.stButton button:hover {background-color: #45a049; border-color: #45a049;}
div[data-testid="stMetricValue"] {font-size: 1.5rem;}
section[data-testid="stSidebar"] > div {padding-top: 2rem;}
</style>
""",
        unsafe_allow_html=True,
    )


# --- Real Inference ---


@st.cache_resource
def load_model_cached():
    """Loads and caches the Keras model to prevent reloading on every run."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "outputs", "instrunet_cnn.keras")

    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}! Please train the model first.")
        return None

    # Check header for HDF5 vs Zip (silent check)
    try:
        with open(model_path, "rb") as f:
            header = f.read(4)

        if header.startswith(b"\x89HDF"):
            # Fallback: Try loading as HDF5 if detected
            try:
                model = load_model(model_path)
                return model
            except Exception:
                pass
    except Exception:
        pass

    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def run_inference(
    filename, y, sr, threshold=0.4, sensitivity=1.0, strategy="Mean", stride_seconds=1.5
):
    model = load_model_cached()
    if model is None:
        return None

    # 1. Preprocessing: Slice Audio into 3-second windows
    window_size = int(3.0 * sr)
    stride = int(stride_seconds * sr)

    windows = []
    # Pad if shorter than window
    if len(y) < window_size:
        pad_width = window_size - len(y)
        windows.append(np.pad(y, (0, pad_width), mode="constant")
    else:
        for start in range(0, len(y) - window_size + 1, stride):
            windows.append(y[start : start + window_size])

    if not windows:
        st.warning("Audio too short to analyze.")
        return None

    # 2. Convert Windows to Spectrogram Images (Batching)
    batch_images = []
    temp_dir = tempfile.mkdtemp()

    progress_bar = st.progress(0, text="Preprocessing windows...")

    for i, win in enumerate(windows):
        temp_path = os.path.join(temp_dir, f"win_{i}.png")
        # Use visualizer helper to match training data exactly
        save_clean_spectrogram(win, sr, temp_path)

        try:
            # Load and resize to 224x224 as expected by the model
            img = image.load_img(temp_path, target_size=(224, 224))
            batch_images.append(image.img_to_array(img))
            os.remove(temp_path)  # Clean up
        except Exception as e:
            print(f"Error processing window {i}: {e}")

        progress_bar.progress(
            (i + 1) / len(windows), text=f"Processed window {i + 1}/{len(windows)}"
        )

    os.rmdir(temp_dir)
    progress_bar.empty()

    if not batch_images:
        st.error("Failed to generate spectrograms.")
        return None

    # 3. Batch Prediction
    batch_tensor = np.array(batch_images)
    # Model outputs logits (linear) -> Apply Softmax to get probabilities
    logits = model.predict(batch_tensor, verbose=0)
    probabilities = tf.nn.softmax(logits).numpy()

    # Capture Timeline Data
    timeline_data = []
    for i in range(len(probabilities)):
        window_start = i * (stride / sr)
        window_end = window_start + (window_size / sr)

        # Create a dict for this window
        win_data = {
            "start_time": float(window_start),
            "end_time": float(window_end),
            "scores": {
                CLASS_NAMES[j]: float(probabilities[i][j])
                for j in range(len(CLASS_NAMES))
            },
        }
        timeline_data.append(win_data)

    # 4. Aggregation Strategy
    if strategy == "Max":
        # Take the maximum confidence across all windows for each class
        final_probs = np.max(probabilities, axis=0)
    else:  # Mean
        # Take the average confidence across all windows
        final_probs = np.mean(probabilities, axis=0)

    # 5. Apply Sensitivity and Format Results
    predictions = []
    for i, code in enumerate(CLASS_NAMES):
        prob = final_probs[i]

        # Sensitivity
        final_conf = prob * sensitivity
        final_conf = min(max(final_conf, 0.0), 1.0)

        predictions.append(
            {
                "instrument": INSTRUMENT_MAP.get(code, code),
                "confidence": float(final_conf),
                "detected": bool(final_conf >= threshold),
            }
        )

    # Sort
    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    result = {
        "metadata": {
            "filename": filename,
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "sample_rate": int(sr),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "predictions": predictions,
        "timeline": timeline_data,
        "threshold": threshold,
        "sensitivity": sensitivity,
        "strategy": strategy,
        "stride_seconds": stride_seconds,
    }
    return result


# --- UI Components ---


def login_page():
    load_css()
    st.title(APP_TITLE)
    st.markdown(
        "<h3 style='text-align: center;'>Secure Login</h3>", unsafe_allow_html=True
    )
    st.write("")

    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            # Form submit button
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                success, msg = login(username, password)
                if success:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("---")
        # Guest Access
        if st.button("👀 Continue as Guest (Demo)", use_container_width=True):
            st.session_state["logged_in"] = True
            st.session_state["username"] = "Guest"
            st.rerun()

        st.write("")
        if st.button("Create New Account", use_container_width=True):
            st.session_state["page"] = "signup"
            st.rerun()

def signup_page():
    load_css()
    st.title(APP_TITLE)
    st.markdown(
        "<h3 style='text-align: center;'>Create Account</h3>", unsafe_allow_html=True
    )
    st.write("")

    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        with st.form("signup_form"):
            username = st.text_input("Choose a Username")
            password = st.text_input("Choose a Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            submitted = st.form_submit_button("Sign Up", use_container_width=True)

            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match.")
                elif not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    success, msg = signup(username, password)
                    if success:
                        st.success(msg)
                        st.session_state["page"] = "login"
                        st.rerun()
                    else:
                        st.error(msg)

    if st.button("Back to Login", use_container_width=True):
        st.session_state["page"] = "login"
        st.rerun()

def dashboard_page():
    load_css()

    # Sidebar
    st.sidebar.markdown(
        f"### <i class='fa-solid fa-circle-user icon-mar'></i> {st.session_state['username']}",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "#### <i class='fa-solid fa-gears icon-mar'></i> Advanced Settings",
        unsafe_allow_html=True,
    )
    threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.4, 0.05)
    sensitivity = st.sidebar.slider("Model Sensitivity", 0.5, 1.5, 1.0, 0.1)

    st.sidebar.markdown(
        "#### <i class='fa-solid fa-clock-rotate-left icon-mar'></i> Temporal Analysis",
        unsafe_allow_html=True,
    )
    stride_seconds = st.sidebar.slider(
        "Time Step (Resolution)",
        0.5,
        3.0,
        1.5,
        0.5,
        help="How often the model makes a prediction. Smaller step = higher detail but slower processing.",
    )
    smoothing_window = st.sidebar.slider(
        "Smoothing Window",
        1,
        5,
        1,
        1,
        help="Moving average window size to smooth out jittery predictions.",
    )
    viz_mode = st.sidebar.radio(
        "Timeline View", ["Line Chart", "Heatmap"], index=0, horizontal=True
    )

    strategy = st.sidebar.selectbox("Aggregation Strategy", ["Mean", "Max"], 0)
    st.sidebar.divider()
    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["prediction_result"] = None
        st.rerun()

    # Main Content
    st.markdown(
        "<h1><i class='fa-solid fa-music icon-mar'></i> Instrunet AI</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #7f8c8d;'>Deep Learning Musical Instrument Detection</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Input Container
    with st.container():
        st.markdown(
            "### <i class='fa-solid fa-folder-open icon-mar'></i> Analysis Input",
            unsafe_allow_html=True,
        )

        tab_upload, tab_record = st.tabs(["📤 Upload File", "🎙️ Record Audio"])

        uploaded_file = None
        with tab_upload:
            file_upload = st.file_uploader(
                "Upload Audio (WAV/MP3)",
                type=["wav", "mp3"],
                label_visibility="collapsed",
            )
            if file_upload:
                uploaded_file = file_upload

        with tab_record:
            # Streamlit 1.34+ audio_input
            recorded_audio = st.audio_input(
                "Record your instrument performance", label_visibility="collapsed"
            )
            if recorded_audio:
                uploaded_file = recorded_audio

        if uploaded_file is not None:
            # Generate a consistent name for recorded files
            fname = getattr(uploaded_file, "name", "recorded_audio.wav")

            if (
                "audio_data" not in st.session_state
                or st.session_state.get("last_file") != fname
            ):
                y, sr = librosa.load(uploaded_file, sr=None)
                st.session_state["audio_data"] = (y, sr)
                st.session_state["last_file"] = fname
                st.session_state["prediction_result"] = None

            y, sr = st.session_state["audio_data"]

            col_audio, col_btn = st.columns([3, 1], vertical_alignment="bottom")
            with col_audio:
                st.audio(uploaded_file)

            run_clicked = False
            with col_btn:
                if st.button("🚀 Identify Instruments", width="stretch"):
                    run_clicked = True

            # Guest Usage Logic
            is_guest = st.session_state.get("username") == "Guest"
            if is_guest:
                usage = st.session_state.get("guest_usage", 0)
                st.caption(f"Guest Usage: {usage}/10")

            if run_clicked:
                if is_guest and st.session_state.get("guest_usage", 0) >= 10:
                    st.error(
                        "🚫 Guest limit reached (10/10). Please create an account to continue."
                    )
                else:
                    with st.spinner("Processing audio & extracting features..."):
                        res = run_inference(
                            fname,
                            y,
                            sr,
                            threshold,
                            sensitivity,
                            strategy,
                            stride_seconds,
                        )
                        if res:
                            st.session_state["prediction_result"] = res
                            st.success("Analysis Complete!")
                            if is_guest:
                                st.session_state["guest_usage"] = (
                                    st.session_state.get("guest_usage", 0) + 1
                                )

    # Visualization (Expander)
    if uploaded_file is not None:
        st.write("")
        with st.expander(
            "📊 Audio Visualization (Waveform & Spectrogram)", expanded=True
        ):
            tab1, tab2 = st.tabs(["Waveform", "Spectrogram"])
            with tab1:
                fig_wav = plot_waveform(y, sr)
                st.pyplot(fig_wav)
                st.session_state["plot_wav_path"] = os.path.join(
                    tempfile.gettempdir(), "temp_wav.png"
                )
                fig_wav.savefig(st.session_state["plot_wav_path"])
            with tab2:
                fig_spec = plot_mel_spectrogram(y, sr)
                st.pyplot(fig_spec)
                st.session_state["plot_spec_path"] = os.path.join(
                    tempfile.gettempdir(), "temp_spec.png"
                )
                fig_spec.savefig(st.session_state["plot_spec_path"])

    # Results Section
    if st.session_state.get("prediction_result"):
        st.markdown("---")
        st.markdown(
            "### <i class='fa-solid fa-bullseye icon-mar'></i> Analysis Results",
            unsafe_allow_html=True,
        )
        res = st.session_state["prediction_result"]

        # Summary Row
        detected = [p["instrument"] for p in res["predictions"] if p["detected"]]

        if detected:
            st.write("#### Detected Instruments:")
            cols = st.columns(min(len(detected), 4))
            for i, inst in enumerate(detected):
                # Display simply without boxes if cols run out
                if i < 4:
                    cols[i].success(f"✅ {inst}")
                else:
                    st.success(f"✅ {inst}")
        else:
            st.info("No distinct instruments detected above threshold.")

        st.write("")

        # Timeline Analysis
        with st.expander("📈 Temporal Timeline Analysis", expanded=True):
            if viz_mode == "Line Chart":
                fig_timeline = plot_timeline_linechart(
                    res["timeline"], smoothing_window=smoothing_window
                )
            else:
                fig_timeline = plot_timeline_heatmap(
                    res["timeline"], smoothing_window=smoothing_window
                )

            st.pyplot(fig_timeline)
            st.session_state["plot_timeline_path"] = os.path.join(
                tempfile.gettempdir(), "temp_timeline.png"
            )
            fig_timeline.savefig(st.session_state["plot_timeline_path"])

        st.write("")

        # Detailed Progress Bars
        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            st.markdown(
                "#### <i class='fa-solid fa-chart-simple icon-mar'></i> Confidence Levels",
                unsafe_allow_html=True,
            )
            for p in res["predictions"]:
                color = "green" if p["detected"] else "gray"
                label = f"{p['instrument']} ({p['confidence'] * 100:.1f}%)"
                st.progress(p["confidence"], text=label)

        with col_res2:
            st.markdown(
                "#### <i class='fa-solid fa-toolbox icon-mar'></i> Actions",
                unsafe_allow_html=True,
            )

            # Everyone can download reports
            json_data = generate_json_report(res)
            st.download_button(
                label="Download JSON Report",
                data=json_data,
                file_name=f"analysis_{uploaded_file.name}.json",
                mime="application/json",
                use_container_width=True,
            )

            pdf_plots = [
                st.session_state.get("plot_wav_path"),
                st.session_state.get("plot_spec_path"),
                st.session_state.get("plot_timeline_path"),
            ]
            # Filter out None values
            pdf_plots = [p for p in pdf_plots if p is not None]

            pdf_bytes = generate_pdf_report(res, plots=pdf_plots)
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"report_{uploaded_file.name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.divider()

            # Saving to Database (Placeholder for future implementation)
            # if st.session_state.get("username") == "Guest": ... (Removed for now)

    # Power User Section (Footer)
    st.write("")
    st.write("")
    with st.expander("🔧 Model Diagnostics (Power User)", expanded=False):
        st.markdown(
            "<p style='font-size: 0.9rem;'>These metrics represent the overall performance of the model on the test dataset.</p>",
            unsafe_allow_html=True,
        )

        # Paths to static assets
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(project_root, "outputs")
        conf_matrix_path = os.path.join(outputs_dir, "normalized_confusion_matrix.png")
        roc_path = os.path.join(outputs_dir, "roc_curves.png")
        report_path = os.path.join(outputs_dir, "classification_report.csv")

        # 1. Classification Report (Table)
        if os.path.exists(report_path):
            st.markdown(
                "#### <i class='fa-solid fa-table-list icon-mar'></i> Global Classification Report",
                unsafe_allow_html=True,
            )
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
                st.image(
                    conf_matrix_path,
                    caption="Normalized Confusion Matrix",
                    width="stretch",
                )
            else:
                st.warning("Confusion Matrix image not found.")

        with col_m2:
            st.subheader("ROC Curves")
            if os.path.exists(roc_path):
                st.image(
                    roc_path,
                    caption="Receiver Operating Characteristic (ROC)",
                    width="stretch",
                )
            else:
                st.warning("ROC Curves image not found.")


# --- Main App Logic ---

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎵", layout="wide")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if st.session_state["logged_in"]:
        dashboard_page()
    else:
        if st.session_state["page"] == "login":
            login_page()
        else:
            signup_page()


if __name__ == "__main__":
    main()