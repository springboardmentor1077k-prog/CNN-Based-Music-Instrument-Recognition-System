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
import smtplib
import re
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import matplotlib
matplotlib.use('Agg')

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
# 2. SESSION STATE INITIALIZATION
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'email' not in st.session_state:
    st.session_state['email'] = ""
if 'last_pdf' not in st.session_state:
    st.session_state['last_pdf'] = None
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False
if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None

def login():
    st.session_state['logged_in'] = True

# ==========================================
# 3. GLOBAL CONSTANTS
# ==========================================
BG_IMAGE_PATH = 'background.jpeg'
LOGO_PATH = 'logo.jpeg'
MODEL_PATH = 'vk_boosted_reg_model.keras'

# --- EMAIL CONFIGURATION ---
try:
    SENDER_EMAIL = st.secrets["email_username"]
    SENDER_PASSWORD = st.secrets["email_password"]
except:
    SENDER_EMAIL = ""
    SENDER_PASSWORD = ""

IMG_HEIGHT = 128
IMG_WIDTH = 128
SR = 16000
CHUNK_DURATION = 3.0

READABLE_NAMES = [
    'Cello', 'Clarinet', 'Flute', 'Ac. Guitar', 'El. Guitar',
    'Organ', 'Piano', 'Saxophone', 'Trumpet', 'Violin', 'Voice'
]

# ==========================================
# 4. ADVANCED CSS STYLING
# ==========================================
def set_style(image_file):
    bg_css = """
    .stApp { background-color: #0E1117; }
    """
    if os.path.exists(image_file):
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        bg_css = f"""
        .stApp {{
            background-color: #0E1117;
            background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        """

    st.markdown(
        f"""
        <style>
        {bg_css}
        
        [data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .main-title {{
            font-family: 'Segoe UI', sans-serif;
            font-size: 3.0rem; 
            font-weight: 700;
            margin-bottom: 5px;
            color: #E0FFFF !important; 
            text-shadow: 0px 0px 10px rgba(0, 201, 255, 0.8), 
                          0px 0px 20px rgba(0, 255, 127, 0.6);
        }}
        
        /* GENERAL TEXT */
        h2, h3, h4, h5, p, span, label, div {{
            color: #FFFFFF !important;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.8);
        }}
        
        /* GLASS INPUT FIELDS */
        .stTextInput input {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 8px !important;
        }}
        .stTextInput input:focus {{
            border-color: #00C9FF !important;
            box-shadow: 0 0 10px rgba(0, 201, 255, 0.3) !important;
        }}
        
        /* RADIO BUTTONS */
        .stRadio label {{
            color: white !important;
            font-weight: bold;
        }}

        /* GLASS BUTTONS */
        div.stButton > button, .download-btn {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            width: 100% !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            display: inline-block !important;
            text-decoration: none !important;
            margin-bottom: 0px !important;
            cursor: pointer !important;
            height: 42px !important;
            margin-top: 0px !important;
        }}
        div.stButton > button:hover, .download-btn:hover {{
            background-color: rgba(0, 201, 255, 0.2) !important;
            border-color: #00C9FF !important;
            box-shadow: 0 0 15px rgba(0, 201, 255, 0.4) !important;
            transform: translateY(-2px);
            color: #E0FFFF !important;
        }}

        [data-testid="stFileUploader"] {{
            background-color: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 10px 15px; 
        }}
        [data-testid="stFileUploader"] section {{ background-color: transparent !important; }}
        
        .stAudio {{
            background-color: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 10px 15px; 
            backdrop-filter: blur(10px);
            margin-top: 10px;
        }}
        
        .login-container, .st-emotion-cache-1r6slb0 {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
        }}
        .login-container {{
            padding: 40px;
            max-width: 400px;
            margin: 100px auto; 
            box-shadow: 0 0 20px rgba(0, 201, 255, 0.2);
        }}
        
        .block-container {{ animation: slideUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1); }}
        @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(50px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        
        /* PROGRESS BARS */
        @keyframes slideIn {{ from {{ width: 0%; }} }}
        .progress-container {{ width: 100%; background-color: rgba(255,255,255,0.1); border-radius: 6px; margin: 8px 0; height: 20px; }}
        .progress-bar {{ height: 100%; border-radius: 6px; text-align: right; padding-right: 10px; line-height: 20px; color: #000; font-weight: bold; font-size: 0.8rem; background: linear-gradient(90deg, #00C9FF, #92FE9D); animation: slideIn 1.5s cubic-bezier(0.65, 0, 0.35, 1) forwards; box-shadow: 0 0 10px rgba(0, 201, 255, 0.5); }}
        .instrument-label {{ font-size: 0.9rem; font-weight: 600; color: #eee; margin-top: 10px; }}
        
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
        .user-email {{ font-size: 0.85rem; color: #aaa !important; }}
        
        @media only screen and (max-width: 600px) {{
            [data-testid="column"] {{ width: 100% !important; flex: 1 1 auto !important; min-width: 100% !important; }}
            .block-container {{ padding-top: 2rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 5. HELPER FUNCTIONS
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

def analyze_timeline(model, y, sr, aggregation_method="Mean"):
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
    
    if not all_preds: 
        return np.zeros(len(READABLE_NAMES)), pd.DataFrame()
    
    all_preds_np = np.array(all_preds)
    
    if aggregation_method == "Mean":
        final_scores = np.mean(all_preds_np, axis=0)
    elif aggregation_method == "Max":
        final_scores = np.max(all_preds_np, axis=0)
    elif aggregation_method == "Voting":
        final_scores = np.mean(all_preds_np > 0.5, axis=0)
    else:
        final_scores = np.mean(all_preds_np, axis=0)

    return final_scores, pd.DataFrame(timeline)

def compute_spectrogram_for_viz(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, fmax=8000)
    return librosa.power_to_db(mel_spec, ref=np.max)

def create_download_link(data, filename):
    json_str = json.dumps(data, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}" class="download-btn">📥 Download JSON</a>'

# --- UPDATED PDF GENERATOR (Fixing Unicode Error) ---
def create_pdf_report(filename, duration, detected_instruments, spec_db, agg_method):
    class PDF(FPDF):
        def header(self):
            # --- 1. CIRCULAR LOGO ---
            if os.path.exists(LOGO_PATH):
                from PIL import Image, ImageDraw
                try:
                    # Open image and convert to RGB
                    img = Image.open(LOGO_PATH).convert("RGB")
                    
                    # Create circular mask
                    mask = Image.new("L", img.size, 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((0, 0) + img.size, fill=255)
                    
                    # Apply mask
                    output = Image.new("RGBA", img.size, (255, 255, 255, 0))
                    output.paste(img, (0, 0), mask)
                    
                    # Save temp file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_logo:
                        output.save(temp_logo.name, format="PNG")
                        # Add to PDF (Top-Left corner)
                        self.image(temp_logo.name, 10, 8, 25)
                except Exception:
                    pass # Skip logo if it fails

            # --- 2. TITLE & LINE ---
            self.set_font('Arial', 'B', 20)
            self.cell(0, 15, 'InstruNetAI Analysis Report', 0, 1, 'C')
            
            # Draw Horizontal Line
            self.set_line_width(0.5)
            self.line(10, 35, 200, 35) # x1, y1, x2, y2
            self.ln(15) # Add spacing after line

        def footer(self):
            # --- 3. PAGE BORDER & FOOTER ---
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
            # Draw Thin Border around the page
            self.set_line_width(0.3)
            self.rect(5, 5, 200, 287) 

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # --- SECTION 1: DETAILS ---
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, "  1. Audio File Details", 0, 1, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(50, 8, "Filename:", 0, 0)
    pdf.cell(0, 8, f"{filename}", 0, 1)
    
    pdf.cell(50, 8, "Duration:", 0, 0)
    pdf.cell(0, 8, f"{duration:.2f} seconds", 0, 1)
    
    pdf.cell(50, 8, "Aggregation Method:", 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, f"{agg_method.upper()}", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    
    # --- SECTION 2: INSTRUMENTS ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "  2. Instruments Detected", 0, 1, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", size=12)
    if detected_instruments:
        for name, score in detected_instruments:
            # FIX: Use a dash '-' instead of unicode bullet point
            pdf.set_text_color(0, 150, 0) 
            pdf.cell(10, 8, "-", 0, 0) # Changed from "●" to "-"
            pdf.set_text_color(0, 0, 0) 
            pdf.cell(0, 8, f"{name} ({int(score*100)}% Confidence)", 0, 1)
    else:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 8, "No instruments detected above threshold.", 0, 1)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # --- SECTION 3: SPECTROGRAM ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "  3. Spectral Analysis", 0, 1, fill=True)
    pdf.ln(5)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_db, aspect='auto', origin='lower', cmap='turbo')
        plt.axis('off')
        plt.savefig(tmpfile.name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        pdf.image(tmpfile.name, x=15, w=180)
    pdf.ln(5)
    
    # --- SECTION 4: SUMMARY ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "  4. Analysis Summary", 0, 1, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", size=11)
    if detected_instruments:
        names = [n for n, s in detected_instruments]
        inst_list = ", ".join(names)
        explanation = (
            f"The InstruNetAI model analyzed the spectral signature of '{filename}' using the "
            f"'{agg_method}' aggregation technique. It successfully identified polyphonic patterns "
            f"matching the following instruments: {inst_list}.\n\n"
            f"The spectrogram visualization above highlights the frequency intensity over time, where "
            f"dominant energy clusters correspond to the identified instrument classes."
        )
    else:
        explanation = (
            f"The InstruNetAI model analyzed the spectral signature of '{filename}' using the "
            f"'{agg_method}' aggregation technique but did not detect specific instrument classes "
            f"within the current confidence threshold."
        )
    
    pdf.multi_cell(0, 6, explanation)
    
    # Encode with 'latin-1' and ignore errors just in case
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- EMAIL SENDER ---
def send_email_with_attachments(recipient_email, pdf_bytes, json_data, filename):
    if "your_email" in SENDER_EMAIL: return False, "⚠️ Email not configured!"
    
    try:
        domain = recipient_email.split('@')[1]
        socket.gethostbyname(domain)
    except Exception:
        return False, "❌ Enter valid email (Domain not found)"

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = f"InstruNetAI Analysis Report: {filename} 🎵"
        body = f"""Hello {st.session_state['username']},

Thank you for using InstruNetAI! 🎧

We have successfully completed the polyphonic analysis of your audio file: "{filename}".

Please find the detailed reports attached to this email:

1. 📄 *PDF Report:* A visual summary containing detected instruments, confidence scores, and spectral analysis.
2. 📊 *JSON Data:* The complete raw analysis file containing the precise timeline data and granular probability scores for every second of the track.

We hope this analysis helps you in your musical journey!

Best regards,
The InstruNetAI Team
"""
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        part_pdf = MIMEBase('application', 'pdf')
        part_pdf.set_payload(pdf_bytes)
        encoders.encode_base64(part_pdf)
        part_pdf.add_header('Content-Disposition', f"attachment; filename=InstruNet_Report.pdf")
        msg.attach(part_pdf)

        # Attach JSON
        json_str = json.dumps(json_data, indent=4)
        part_json = MIMEBase('application', 'json')
        part_json.set_payload(json_str)
        encoders.encode_base64(part_json)
        part_json.add_header('Content-Disposition', f"attachment; filename={filename}_analysis.json")
        msg.attach(part_json)
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        return True, "✅ Email sent successfully!"
    
    except smtplib.SMTPRecipientsRefused:
        return False, "❌ Enter valid email (Rejected)"
    except Exception as e:
        return False, f"❌ Email failed: {str(e)}"

# ==========================================
# 6. VISUALIZATIONS
# ==========================================
def plot_spectrogram(spec_db, sr):
    fig = px.imshow(spec_db, labels=dict(x="Time", y="Hz", color="dB"), y=librosa.mel_frequencies(n_mels=256, fmax=8000), origin='lower', aspect='auto', color_continuous_scale='Turbo')
    fig.update_layout(title="<b>🎵 Spectral Analysis</b>", margin=dict(l=0, r=0, t=30, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

def plot_timeline(timeline_df, threshold):
    melted = timeline_df.melt(id_vars=['Start (s)', 'End (s)'], var_name='Instrument', value_name='Score')
    melted = melted[melted['Score'] >= threshold]
    if melted.empty: return None
    fig = px.density_heatmap(melted, x="Start (s)", y="Instrument", z="Score", nbinsx=len(timeline_df), color_continuous_scale='Plasma', range_color=[threshold, 1.0], title="<b>⏱️ Timeline Activity</b>")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

# --- NEW: WAVEFORM PLOT FUNCTION ---
def plot_waveform(y, sr):
    # Downsample for speed (plot every 100th point)
    step = max(1, len(y) // 5000) 
    y_downsampled = y[::step]
    x_axis = np.linspace(0, len(y) / sr, len(y_downsampled))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, 
        y=y_downsampled, 
        mode='lines', 
        name='Amplitude',
        line=dict(color='#00C9FF', width=1.5) # Cyan color to match reference
    ))
    
    fig.update_layout(
        title="<b>🌊 Audio Waveform</b>",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        margin=dict(l=0, r=0, t=30, b=0),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        yaxis=dict(range=[-1.1, 1.1])
    )
    return fig
# -----------------------------------

# ==========================================
# 7. LOGIN PAGE
# ==========================================
def login_page():
    logo_b64 = ""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    
    header_html = f"""<div style="display:flex;align-items:center;justify-content:center;flex-wrap:wrap;background-color:rgba(0,0,0,0.5);padding:20px 40px;border-radius:15px;margin-bottom:30px;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.15);box-shadow:0 0 20px rgba(0,201,255,0.2);width:fit-content;max-width:90%;margin-left:auto;margin-right:auto;"><img src="data:image/jpeg;base64,{logo_b64}" style="border-radius:50%;object-fit:cover;margin-right:20px;border:2px solid rgba(255,255,255,0.2);width:clamp(40px,10vw,60px);height:clamp(40px,10vw,60px);"><h1 style="font-family:'Segoe UI',sans-serif;font-weight:700;margin:0;color:#E0FFFF;text-shadow:0px 0px 10px rgba(0,201,255,0.8),0px 0px 20px rgba(0,255,127,0.6);font-size:clamp(1.5rem,5vw,2.5rem);">InstruNetAI</h1></div>"""

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(header_html, unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; color: #E0FFFF;">Welcome to InstruNetAI</h2>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            user = st.text_input("Username", placeholder="Enter your name")
            mail = st.text_input("Email Address", placeholder="Enter your email")
            submitted = st.form_submit_button("Login to Dashboard", use_container_width=True)
            
            if submitted:
                email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                if not user or not mail:
                    st.error("Please fill in both fields.")
                elif not re.match(email_pattern, mail):
                    st.error("⚠️ Please enter a valid email")
                else:
                    st.session_state['username'] = user
                    st.session_state['email'] = mail
                    login()
                    try: st.rerun()
                    except AttributeError: st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 8. MAIN LAYOUT
# ==========================================
set_style(BG_IMAGE_PATH)

if not st.session_state['logged_in']:
    login_page()
    st.stop()

col_logo, col_title = st.columns([1, 6])
with col_logo:
    try: st.image(LOGO_PATH, use_container_width=True)
    except: pass
with col_title:
    st.markdown('<h1 class="main-title">InstruNetAI Dashboard</h1>', unsafe_allow_html=True)
    st.caption("Advanced Polyphonic Audio Recognition System")

with st.sidebar:
    st.markdown(f'<div class="user-profile"><div class="user-name">{st.session_state["username"]}</div><div class="user-email">{st.session_state["email"]}</div></div>', unsafe_allow_html=True)
    st.header("🎛️ Settings")
    
    aggregation_method = st.selectbox(
        "Aggregation Method",
        ["Mean", "Max", "Voting"],
        index=0,
        help="Mean: Average confidence.\nMax: Highest peak detection.\nVoting: Frequency of occurrence."
    )

    threshold = st.slider("Detection Sensitivity", 0.0, 1.0, 0.30, 0.05)
    st.markdown("---")
    st.markdown("**System Status:**\n- 🟢 Model: Loaded\n- 🟢 GPU: Active\n- 🟢 Mode: Multi-Label")

uploaded_file = st.file_uploader("📂 Upload Audio Track (WAV/MP3)", type=['wav', 'mp3'])

if uploaded_file:
    st.audio(uploaded_file)
    
    if st.session_state.get('last_uploaded_file') != uploaded_file.name:
        st.session_state['analysis_complete'] = False
        st.session_state['last_uploaded_file'] = uploaded_file.name

    analyze_btn = st.button("▶️ Analyze Audio", use_container_width=True)
    
    if analyze_btn:
        st.session_state['analysis_complete'] = True

    if st.session_state['analysis_complete']:
        model = load_model()
        if model:
            uploaded_file.seek(0)
            
            status_text = st.empty()
            progress_bar = st.progress(0)
            status_text.text("⏳ Reading Audio...")
            y, sr = librosa.load(uploaded_file, sr=SR, mono=True)
            y = librosa.util.normalize(y)
            progress_bar.progress(30)
            
            status_text.text(f"🧠 Neural Network Processing ({aggregation_method})...")
            avg_preds, timeline_df = analyze_timeline(model, y, sr, aggregation_method)
            progress_bar.progress(70)
            
            status_text.text("🎨 Creating Visuals...")
            spec_db = compute_spectrogram_for_viz(y, sr)
            progress_bar.progress(100)
            
            # --- AUTO GENERATE PDF ---
            data_filtered = sorted(zip(READABLE_NAMES, avg_preds), key=lambda x: x[1], reverse=True)
            filtered = [(n, s) for n, s in data_filtered if s >= threshold]
            pdf_bytes = create_pdf_report(uploaded_file.name, len(y)/sr, filtered, spec_db, aggregation_method)
            st.session_state['last_pdf'] = pdf_bytes
            # -------------------------

            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            col1, col2 = st.columns([2, 1])
            with col1:
                # 1. Spectrogram
                st.plotly_chart(plot_spectrogram(spec_db, sr), use_container_width=True)
                
                # 2. Timeline
                t_fig = plot_timeline(timeline_df, threshold)
                if t_fig: st.plotly_chart(t_fig, use_container_width=True)
                else: st.info("No clear timeline activity.")
                
                # 3. Waveform (NEW: ADDED HERE)
                st.plotly_chart(plot_waveform(y, sr), use_container_width=True)

            with col2:
                st.markdown("### 🎹 Results")
                if filtered:
                    for name, score in filtered:
                        pct = int(score * 100)
                        st.markdown(f'<div class="instrument-label">{name}</div><div class="progress-container"><div class="progress-bar" style="width: {pct}%;">{pct}%</div></div>', unsafe_allow_html=True)
                    st.markdown("---")
                    st.success(f"**Identified:** {', '.join([n for n, s in filtered])}")
                else:
                    st.warning("No instruments detected.")
                
                with st.expander("📝 Instrument Details", expanded=False):
                    detected_names = set([n for n, s in filtered]) if filtered else set()
                    html_parts = ['<div style="background-color:rgba(0,0,0,0.5);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.15);border-radius:12px;padding:20px;margin-top:10px;">']
                    for name in sorted(READABLE_NAMES):
                        if name in detected_names:
                            icon, style = "✔", "background-color:#2ecc71;box-shadow:0 0 8px rgba(46,204,113,0.6);"
                            txt = "color:#2ecc71;font-weight:bold;"
                        else:
                            icon, style = "✖", "background-color:#e74c3c;box-shadow:0 0 8px rgba(231,76,60,0.6);"
                            txt = "color:#bbb;"
                        html_parts.append(f'<div style="display:flex;align-items:center;margin-bottom:12px;"><div style="width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;margin-right:15px;{style}">{icon}</div><span style="font-size:1rem;{txt}">{name}</span></div>')
                    html_parts.append("</div>")
                    st.markdown("".join(html_parts), unsafe_allow_html=True)

                st.markdown("---")
                st.write("### 📄 Export Reports")
                
                c1, c2 = st.columns(2)
                with c1:
                    report = {"file": uploaded_file.name, "duration": len(y)/sr, "aggregation": aggregation_method, "detected": [n for n, s in filtered], "timeline": timeline_df.to_dict(orient='records')}
                    st.markdown(create_download_link(report, "instrunet_analysis.json"), unsafe_allow_html=True)

                    if st.session_state['last_pdf']:
                        b64_pdf = base64.b64encode(st.session_state['last_pdf']).decode('utf-8')
                        st.markdown(f'<a href="data:application/pdf;base64,{b64_pdf}" download="InstruNet_Report.pdf" class="download-btn">📥 Download PDF</a>', unsafe_allow_html=True)

                with c2:
                    st.markdown('<div style="font-weight:bold; margin-bottom:5px;">Share Report via Email:</div>', unsafe_allow_html=True)
                    
                    email_mode = st.radio("Send to:", ["Registered Email", "New Recipient"], horizontal=True, label_visibility="collapsed")
                    
                    if email_mode == "Registered Email":
                        target_email = st.session_state['email']
                        st.text_input("Recipient:", value=target_email, disabled=True, label_visibility="collapsed")
                    else:
                        target_email = st.text_input("Recipient:", placeholder="Enter recipient email...", label_visibility="collapsed")

                    if st.button("Share via Email ➔", key="email_btn"):
                        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                        
                        if not target_email or not re.match(email_pattern, target_email):
                            st.markdown(f'<div style="background-color:rgba(231,76,60,0.2);border:1px solid #e74c3c;border-radius:5px;padding:5px 10px;color:#e74c3c;font-weight:bold;text-align:center;margin-top:5px;font-size:0.9rem;">⚠️ Invalid Email Format</div>', unsafe_allow_html=True)
                        elif st.session_state['last_pdf']:
                            with st.spinner("Sending..."):
                                success, msg = send_email_with_attachments(target_email, st.session_state['last_pdf'], report, uploaded_file.name)
                                if success:
                                    st.markdown(f'<div style="background-color:rgba(46,204,113,0.2);border:1px solid #2ecc71;border-radius:5px;padding:5px 10px;color:#2ecc71;font-weight:bold;text-align:center;margin-top:5px;font-size:0.9rem;">{msg}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div style="background-color:rgba(231,76,60,0.2);border:1px solid #e74c3c;border-radius:5px;padding:5px 10px;color:#e74c3c;font-weight:bold;text-align:center;margin-top:5px;font-size:0.9rem;">{msg}</div>', unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Processing... please wait.")

elif not uploaded_file:
    st.info("👋 Welcome to InstruNet AI! Please upload a file to begin.")
