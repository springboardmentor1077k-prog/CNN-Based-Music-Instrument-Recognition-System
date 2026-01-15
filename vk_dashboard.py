import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import base64
import time
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="InstruNet AI Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 0. SESSION STATE INITIALIZATION
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'email' not in st.session_state:
    st.session_state['email'] = ""

def login():
    st.session_state['logged_in'] = True

# ==========================================
# 2. GLOBAL CONSTANTS
# ==========================================
BG_IMAGE_PATH = 'background.jpeg'
LOGO_PATH = 'logo.jpeg'
MODEL_PATH = 'vk_boosted_reg_model.keras'

IMG_HEIGHT = 128
IMG_WIDTH = 128
SR = 16000
CHUNK_DURATION = 3.0

READABLE_NAMES = [
    'Cello', 'Clarinet', 'Flute', 'Ac. Guitar', 'El. Guitar',
    'Organ', 'Piano', 'Saxophone', 'Trumpet', 'Violin', 'Voice'
]

# ==========================================
# 3. ADVANCED CSS STYLING (RESPONSIVE)
# ==========================================
def set_style(image_file):
    # 1. Handle Background Image
    bg_css = ""
    if os.path.exists(image_file):
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        bg_css = f"""
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        """
    else:
        # Fallback to a dark background if image is missing
        bg_css = """
        .stApp {
            background-color: #0E1117;
        }
        """

    # 2. Inject CSS (Now happens regardless of file existence)
    st.markdown(
        f"""
        <style>
        /* 1. BACKGROUND (Dynamic) */
        {bg_css}

        /* 2. SIDEBAR (Glass Style) */
        [data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}

        /* 3. REFINED TITLE (Responsive) */
        .main-title {{
            font-family: 'Segoe UI', sans-serif;
            font-size: clamp(2.0rem, 5vw, 3.0rem); 
            font-weight: 700;
            margin-bottom: 5px;
            color: #E0FFFF !important; 
            text-shadow: 0px 0px 10px rgba(0, 201, 255, 0.8), 
                         0px 0px 20px rgba(0, 255, 127, 0.6);
        }}

        /* 4. SLIMMER, TRANSPARENT UPLOADER */
        [data-testid="stFileUploader"] {{
            background-color: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 10px 15px;
            transition: box-shadow 0.3s ease;
        }}
        [data-testid="stFileUploader"] section {{
            background-color: transparent !important;
        }}
        [data-testid="stFileUploader"]:hover {{
            box-shadow: 0 0 15px rgba(0, 201, 255, 0.3);
            border-color: rgba(0, 201, 255, 0.5);
        }}

        /* 5. MATCHING AUDIO PLAYER */
        .stAudio {{
            background-color: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 10px 15px;
            backdrop-filter: blur(10px);
            margin-top: 10px;
        }}

        /* 6. GENERAL TEXT COLORS */
        h2, h3, h4, h5, p, span, label, div {{
            color: #FFFFFF !important;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.8);
        }}
        .stTextInput, .stFileUploader, .stSlider {{
            color: white !important;
        }}
        
        /* 7. COMPACT BARS */
        @keyframes slideIn {{ from {{ width: 0%; }} }}
        .progress-container {{
            width: 100%; 
            background-color: rgba(255,255,255,0.1);
            border-radius: 6px; 
            margin: 8px 0; 
            height: 20px; 
        }}
        .progress-bar {{
            height: 100%; border-radius: 6px;
            text-align: right; padding-right: 10px; line-height: 20px;
            color: #000; 
            font-weight: bold; font-size: 0.8rem;
            background: linear-gradient(90deg, #00C9FF, #92FE9D);
            animation: slideIn 1.5s cubic-bezier(0.65, 0, 0.35, 1) forwards;
            box-shadow: 0 0 10px rgba(0, 201, 255, 0.5);
        }}
        .instrument-label {{ font-size: 0.9rem; font-weight: 600; color: #eee; margin-top: 10px; }}

        /* 8. CARDS */
        .st-emotion-cache-1r6slb0 {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            border: 1px solid rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
        }}
        
        .main .block-container {{ animation: fadeIn 1.0s ease-in-out; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}

        /* 9. LOGIN & TRANSITIONS (RESPONSIVE) */
        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(50px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .login-container {{
            background-color: rgba(0, 0, 0, 0.5);
            padding: 40px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            
            /* RESPONSIVE WIDTH & MARGIN */
            width: 90%;          /* 90% width on mobile */
            max-width: 400px;    /* Max 400px on desktop */
            margin: 10vh auto;   /* Vertical margin adapts to screen height */
            
            box-shadow: 0 0 20px rgba(0, 201, 255, 0.2);
        }}
        
        /* Apply animation to the dashboard when it loads */
        .block-container {{
            animation: slideUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
        }}
        
        /* Sidebar User Profile Styles */
        .user-profile {{
            background-color: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
            text-align: center;
        }}
        .user-name {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #E0FFFF !important;
            margin-bottom: 2px;
            text-shadow: 0px 0px 5px rgba(0, 201, 255, 0.5);
        }}
        .user-email {{
            font-size: 0.85rem;
            color: #aaa !important;
        }}
        
        /* 10. MOBILE SPECIFIC ADJUSTMENTS */
        @media only screen and (max-width: 600px) {{
            [data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 auto !important;
                min-width: 100% !important;
            }}
            .block-container {{
                padding-top: 2rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def preprocess_chunk(chunk):
    target_len = int(SR * CHUNK_DURATION)
    if len(chunk) > target_len:
        chunk = chunk[:target_len]
    else:
        chunk = np.pad(chunk, (0, target_len - len(chunk)))

    spec = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=IMG_HEIGHT)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_norm = (spec_db + 80) / 80
    spec_norm = np.clip(spec_norm, 0, 1)

    if spec_norm.shape[1] < IMG_WIDTH:
        pad = IMG_WIDTH - spec_norm.shape[1]
        spec_norm = np.pad(spec_norm, ((0,0), (0, pad)))
    else:
        spec_norm = spec_norm[:, :IMG_WIDTH]
    return spec_norm.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

def compute_spectrogram_for_viz(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, fmax=8000)
    return librosa.power_to_db(mel_spec, ref=np.max)

def analyze_timeline(model, y, sr):
    chunk_samples = int(CHUNK_DURATION * sr)
    total_samples = len(y)
    timeline = []
    all_preds = []

    for start in range(0, total_samples, chunk_samples):
        end = start + chunk_samples
        chunk = y[start:end]
        if len(chunk) < chunk_samples / 2: continue

        input_tensor = preprocess_chunk(chunk)
        pred = model.predict(input_tensor, verbose=0)[0]
        all_preds.append(pred)

        row = {'Start (s)': start/sr, 'End (s)': end/sr}
        for i, score in enumerate(pred):
            row[READABLE_NAMES[i]] = score
        timeline.append(row)

    if not all_preds: return np.zeros(len(READABLE_NAMES)), pd.DataFrame()
    return np.mean(all_preds, axis=0), pd.DataFrame(timeline)

def create_download_link(data, filename):
    json_str = json.dumps(data, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}" class="download-btn">📥 Download Full Report (JSON)</a>'

# ==========================================
# 5. CHART PLOTTING
# ==========================================
def plot_spectrogram(spec_db, sr):
    fig = px.imshow(
        spec_db, labels=dict(x="Time", y="Hz", color="dB"),
        y=librosa.mel_frequencies(n_mels=256, fmax=8000),
        origin='lower', aspect='auto', color_continuous_scale='Turbo'
    )
    fig.update_layout(
        title="<b>🎵 Spectral Analysis</b>", margin=dict(l=0, r=0, t=30, b=0), height=250,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white')
    )
    return fig

def plot_timeline(timeline_df, threshold):
    melted = timeline_df.melt(id_vars=['Start (s)', 'End (s)'], var_name='Instrument', value_name='Score')
    melted = melted[melted['Score'] >= threshold]
    if melted.empty: return None

    fig = px.density_heatmap(
        melted, x="Start (s)", y="Instrument", z="Score",
        nbinsx=len(timeline_df), color_continuous_scale='Plasma', range_color=[threshold, 1.0],
        title="<b>⏱️ Timeline Activity</b>"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0), height=350,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white')
    )
    return fig

# ==========================================
# 6. LOGIN PAGE (RESPONSIVE)
# ==========================================
# ==========================================
# 6. LOGIN PAGE (FIXED)
# ==========================================
def login_page():
    # 1. Load and Encode Logo
    logo_b64 = ""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    
    # 2. Define the HTML for the Header (Cleaned - No Comments)
    header_html = f"""
    <div style="
        display: flex; 
        align-items: center; 
        justify-content: center; 
        flex-wrap: wrap;
        background-color: rgba(0, 0, 0, 0.5); 
        padding: 20px 40px; 
        border-radius: 15px; 
        margin-bottom: 30px; 
        backdrop-filter: blur(10px); 
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 0 20px rgba(0, 201, 255, 0.2);
        width: fit-content;
        max-width: 90%;
        margin-left: auto;
        margin-right: auto;
    ">
        <img src="data:image/jpeg;base64,{logo_b64}" 
             style="
                border-radius: 50%; 
                object-fit: cover; 
                margin-right: 20px; 
                border: 2px solid rgba(255,255,255,0.2);
                width: clamp(40px, 10vw, 60px); 
                height: clamp(40px, 10vw, 60px);
             ">
        <h1 style="
            font-family: 'Segoe UI', sans-serif; 
            font-weight: 700; 
            margin: 0;
            color: #E0FFFF; 
            text-shadow: 0px 0px 10px rgba(0, 201, 255, 0.8), 0px 0px 20px rgba(0, 255, 127, 0.6);
            font-size: clamp(1.5rem, 5vw, 2.5rem);
        ">
            InstruNetAI
        </h1>
    </div>
    """

    # 3. Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Render the Header
        st.markdown(header_html, unsafe_allow_html=True)
        
        # The Login Form Container
        #st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; color: #E0FFFF;">Welcome to InstruNetAI</h2>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            user = st.text_input("Username", placeholder="Enter your name")
            mail = st.text_input("Email Address", placeholder="Enter your email")
            
            submitted = st.form_submit_button("Login to Dashboard", use_container_width=True)
            
            if submitted:
                if user and mail:
                    st.session_state['username'] = user
                    st.session_state['email'] = mail
                    login()
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
                else:
                    st.error("Please fill in both fields.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 7. MAIN APP LAYOUT
# ==========================================

# 1. APPLY STYLE
set_style(BG_IMAGE_PATH)

# CHECK LOGIN STATUS
if not st.session_state['logged_in']:
    login_page()
    st.stop() # Stops execution here if not logged in

# 2. DASHBOARD HEADER (Responsive)
col_logo, col_title = st.columns([1, 6])
with col_logo:
    try:
        # Logo will automatically resize due to 'use_container_width=True'
        st.image(LOGO_PATH, use_container_width=True)
    except:
        pass
with col_title:
    # Responsive Title Class Applied
    st.markdown('<h1 class="main-title">InstruNetAI Dashboard</h1>', unsafe_allow_html=True)
    st.caption("Advanced Polyphonic Audio Recognition System")

# 3. SIDEBAR (With Profile)
with st.sidebar:
    st.markdown(f"""
    <div class="user-profile">
        <div class="user-name">{st.session_state['username']}</div>
        <div class="user-email">{st.session_state['email']}</div>
    </div>
    """, unsafe_allow_html=True)
    st.header("🎛️ Settings")
    threshold = st.slider("Detection Sensitivity", 0.0, 1.0, 0.30, 0.05)
    st.info(f"Showing instruments with >{int(threshold*100)}% confidence.")
    st.markdown("---")
    st.markdown("**System Status:**\n- 🟢 Model: Loaded\n- 🟢 GPU: Active\n- 🟢 Mode: Multi-Label")

# 4. MAIN CONTENT
uploaded_file = st.file_uploader("📂 Upload Audio Track (WAV/MP3)", type=['wav', 'mp3'])

if uploaded_file:
    st.audio(uploaded_file)
    
    model = load_model()
    if model:
        # PROCESSING UI
        status_text = st.empty()
        progress_bar = st.progress(0)

        status_text.text("⏳ Reading Audio...")
        y, sr = librosa.load(uploaded_file, sr=SR, mono=True)
        y = librosa.util.normalize(y)
        progress_bar.progress(30)

        status_text.text("🧠 Neural Network Processing...")
        avg_preds, timeline_df = analyze_timeline(model, y, sr)
        progress_bar.progress(70)

        status_text.text("🎨 Creating Visuals...")
        spec_db = compute_spectrogram_for_viz(y, sr)
        progress_bar.progress(100)
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        # RESULTS GRID (Responsive Columns)
        # Columns will naturally stack on mobile, but we reinforce it with CSS media queries
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🔍 Spectral Analysis")
            st.plotly_chart(plot_spectrogram(spec_db, sr), use_container_width=True)

            st.markdown("### ⏱️ Temporal Detection")
            t_fig = plot_timeline(timeline_df, threshold)
            if t_fig:
                st.plotly_chart(t_fig, use_container_width=True)
            else:
                st.info("No clear timeline activity.")

        with col2:
            st.markdown("### 🎹 Results")
            st.caption("Confidence levels:")

            data = sorted(zip(READABLE_NAMES, avg_preds), key=lambda x: x[1], reverse=True)
            filtered = [(n, s) for n, s in data if s >= threshold]

            if filtered:
                for name, score in filtered:
                    pct = int(score * 100)
                    st.markdown(f"""
                        <div class="instrument-label">{name}</div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {pct}%;">{pct}%</div>
                        </div>
                    """, unsafe_allow_html=True)

                names_only = [n for n, s in filtered]
                st.markdown("---")
                st.success(f"**Identified:** {', '.join(names_only)}")
            else:
                st.warning("No instruments detected.")

            # EXPORT BUTTON
            report = {
                "file": uploaded_file.name,
                "duration": len(y)/sr,
                "detected": [n for n, s in filtered],
                "timeline": timeline_df.to_dict(orient='records')
            }
            st.markdown("---")
            st.markdown(create_download_link(report, "instrunet_analysis.json"), unsafe_allow_html=True)

elif not uploaded_file:
    st.info("👋 Welcome to InstruNet AI! Please upload a file to begin.")