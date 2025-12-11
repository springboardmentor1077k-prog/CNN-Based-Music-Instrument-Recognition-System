import streamlit as st
import os

# --- CONFIGURATION ---
AUGMENTED_FOLDER = './Augmented_Samples'

st.set_page_config(page_title="Augmentation Demo", layout="wide")
st.title("🎧 Audio Augmentation Comparison")
st.markdown("Listen to the difference between the Original audio and the Augmented versions.")

if not os.path.exists(AUGMENTED_FOLDER):
    st.error(f"Folder '{AUGMENTED_FOLDER}' not found. Did you run perform_augmentation.py?")
    st.stop()

# Get list of all files
all_files = os.listdir(AUGMENTED_FOLDER)
# Filter for just the 'original' files to use as headers
originals = [f for f in all_files if '_original.wav' in f]

for orig in originals:
    st.divider()
    
    # Extract the base name (e.g., "[pia]01_original.wav" -> "[pia]01")
    base_name = orig.replace('_original.wav', '')
    
    st.subheader(f"🎵 Sample: {base_name}")
    
    # Create columns for side-by-side playback
    col1, col2, col3, col4 = st.columns(4)
    
    # 1. Original
    with col1:
        st.write("**1. Original**")
        st.audio(os.path.join(AUGMENTED_FOLDER, orig))

    # 2. Pitch Shift
    with col2:
        pitch_file = f"{base_name}_pitch_up.wav"
        if pitch_file in all_files:
            st.write("**2. Pitch Shift (+2)**")
            st.audio(os.path.join(AUGMENTED_FOLDER, pitch_file))

    # 3. Noise
    with col3:
        noise_file = f"{base_name}_noise.wav"
        if noise_file in all_files:
            st.write("**3. Added Noise**")
            st.audio(os.path.join(AUGMENTED_FOLDER, noise_file))
            
    # 4. Time Shift
    with col4:
        shift_file = f"{base_name}_shifted.wav"
        if shift_file in all_files:
            st.write("**4. Time Shift**")
            st.audio(os.path.join(AUGMENTED_FOLDER, shift_file))