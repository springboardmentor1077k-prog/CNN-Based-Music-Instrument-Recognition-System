import streamlit as st
import tempfile
import json
from datetime import datetime

from utils.api_client import analyze_audio
from utils.visualizer import plot_spectrogram

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .app-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }
    
    .app-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .section-card {
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #eaeaea;
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        color: #2d3748;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #f0f0f0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    
    /* Instrument items */
    .instrument-card {
        background: linear-gradient(135deg, #f6f8ff 0%, #f9f9f9 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .instrument-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .instrument-name {
        color: #2d3748;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .confidence-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    .confidence-text {
        color: #4a5568;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .present-badge {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .absent-badge {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.8rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    .param-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Intensity graph */
    .intensity-container {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Upload area */
    .upload-area {
        background: linear-gradient(135deg, #f6f8ff 0%, #f0f4ff 100%);
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #f0f4ff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Divider styling */
    .stDivider {
        border-color: #e2e8f0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.9rem;
        padding: 1rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide"
)

# ------------------------------------------------------------
# SAFE HELPERS
# ------------------------------------------------------------
def safe_get(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None

def normalize_predictions(results):
    """
    Converts ANY backend response into:
    [
        {"instrument": str, "confidence": float}
    ]
    """
    preds = []

    if not isinstance(results, dict):
        return preds

    # Case 1: top_predictions (ideal)
    if "top_predictions" in results:
        for p in results["top_predictions"]:
            preds.append({
                "instrument": safe_get(p, "instrument", "label", "class", "name"),
                "confidence": float(safe_get(p, "confidence", "score", "prob", 0))
            })

    # Case 2: single prediction
    elif "prediction" in results:
        preds.append({
            "instrument": results.get("prediction"),
            "confidence": float(results.get("confidence", 0))
        })

    # Case 3: generic predictions list
    elif "predictions" in results:
        for p in results["predictions"]:
            if isinstance(p, dict):
                preds.append({
                    "instrument": safe_get(p, "instrument", "label", "class", "name"),
                    "confidence": float(safe_get(p, "confidence", "score", "prob", 0))
                })
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                preds.append({
                    "instrument": p[0],
                    "confidence": float(p[1])
                })

    # Clean invalid rows
    preds = [
        p for p in preds
        if p["instrument"] is not None
    ]

    return preds

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">🎵 InstruNet AI</div>
    <div class="app-subtitle">Upload → Analyze → Discover Instruments</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR PARAMETERS
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ **Analysis Parameters**")
    
    with st.container():
        st.markdown('<div class="param-card">', unsafe_allow_html=True)
        segment_length = st.slider(
            "**Segment Length** (seconds)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Length of audio segments for analysis"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="param-card">', unsafe_allow_html=True)
        threshold = st.slider(
            "**Confidence Threshold**",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
            help="Minimum confidence to consider an instrument present"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="param-card">', unsafe_allow_html=True)
        aggregation = st.selectbox(
            "**Aggregation Method**",
            ["mean", "max", "vote"],
            help="Method to combine predictions across segments"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="param-card">', unsafe_allow_html=True)
        top_k = st.slider(
            "**Top-K Instruments**",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of top instruments to display"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 **Analyze Track**", use_container_width=True)

# ------------------------------------------------------------
# MAIN LAYOUT
# ------------------------------------------------------------
left, right = st.columns([1, 1])

# ------------------------------------------------------------
# LEFT COLUMN - UPLOAD SECTION
# ------------------------------------------------------------
with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload Audio</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "**Drag & drop or browse**",
        type=["wav", "mp3", "flac", "ogg"],
        label_visibility="collapsed",
        help="Supported formats: wav, mp3, flac, ogg"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    audio_path = None

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        st.markdown("**Audio Preview**")
        st.audio(audio_path)
        st.success("✅ Audio file uploaded successfully!")
    else:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("**No audio uploaded** - Please upload a file to begin analysis")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# RIGHT COLUMN - ANALYSIS RESULTS
# ------------------------------------------------------------
with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("**Run analysis** to see results - Upload audio and click 'Analyze Track'")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Show some analysis stats if results exist
        results = st.session_state.get("results")
        if results:
            st.success("✅ Analysis completed successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# RUN ANALYSIS
# ------------------------------------------------------------
if analyze_btn:
    if not audio_path:
        st.error("❌ **Please upload an audio file first**")
        st.stop()

    with st.spinner("🎵 **Analyzing audio with AI...**"):
        try:
            st.session_state.results = analyze_audio(
                audio_path,
                segment_length=segment_length,
                aggregation=aggregation
            )
            st.success("✅ **Analysis completed successfully!**")
            st.rerun()

        except Exception as e:
            st.error("❌ **Backend error occurred**")
            st.exception(e)
            st.stop()

# ------------------------------------------------------------
# DETECTED INSTRUMENTS SECTION
# ------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🎸 Detected Instruments</div>', unsafe_allow_html=True)

results = st.session_state.get("results")
preds = normalize_predictions(results)

if not preds:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.warning("No valid predictions returned by backend")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    cols = st.columns(2)
    for idx, p in enumerate(preds[:top_k]):
        with cols[idx % 2]:
            is_present = p["confidence"] >= threshold
            st.markdown(f'<div class="instrument-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f'<div class="instrument-name">{p["instrument"]}</div>', unsafe_allow_html=True)
            with col2:
                if is_present:
                    st.markdown('<span class="present-badge">PRESENT</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="absent-badge">ABSENT</span>', unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f'<div class="confidence-text">{p["confidence"]*100:.2f}% confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-bar"><div class="confidence-fill" style="width: {p["confidence"]*100}%"></div></div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f"*Showing {len(preds[:top_k])} instruments*")
    
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# INTENSITY GRAPH SECTION
# ------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔥 Instrument Intensity</div>', unsafe_allow_html=True)

if preds:
    st.markdown('<div class="intensity-container">', unsafe_allow_html=True)
    for p in preds[:top_k]:
        if p["confidence"] >= threshold:
            col1, col2, col3 = st.columns([2, 6, 2])
            with col1:
                st.markdown(f'**{p["instrument"]}**')
            with col2:
                st.progress(min(1.0, p["confidence"]))
            with col3:
                st.markdown(f'`{p["confidence"]*100:.1f}%`')
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.info("No intensity data available - Run analysis first")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# SPECTROGRAM SECTION
# ------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🎼 Spectrogram</div>', unsafe_allow_html=True)

if audio_path:
    try:
        fig = plot_spectrogram(audio_path)
        st.pyplot(fig)
        st.caption("Visual representation of audio frequencies over time")
    except Exception as e:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.warning("Could not generate spectrogram")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.info("Upload audio to generate spectrogram")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# EXPORT SECTION
# ------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📦 Export Results</div>', unsafe_allow_html=True)

if results:
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.markdown("**Export your analysis results**")
        st.markdown("*Includes: Detected instruments, confidence scores, timestamps, and metadata*")
    with col2:
        st.download_button(
            "📄 **Export JSON**",
            data=json.dumps(results, indent=2),
            file_name=f"instrunet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    with col3:
        st.button(
            "📊 **View Report**",
            help="Generate detailed analysis report",
            use_container_width=True
        )
else:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.info("Complete analysis to enable export options")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.caption("🎵 **InstruNet AI** • Advanced Music Instrument Recognition System • v1.0")
st.markdown('</div>', unsafe_allow_html=True)