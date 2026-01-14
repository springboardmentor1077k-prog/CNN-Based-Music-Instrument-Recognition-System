import streamlit as st
import os
import json
import hashlib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
USER_DB_FILE = "users.json"
APP_TITLE = "Instrunet AI Dashboard"

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

def plot_mel_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, power=1.0)
    M_db = librosa.amplitude_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram (Amplitude)')
    return fig

# --- UI Components ---

def login_page():
    st.title(APP_TITLE)
    st.header("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Login"):
            success, msg = login(username, password)
            if success:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
                
    with col2:
        if st.button("Go to Signup"):
            st.session_state['page'] = 'signup'
            st.rerun()

def signup_page():
    st.title(APP_TITLE)
    st.header("Sign Up")
    
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Sign Up"):
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
    
    with col2:
        if st.button("Back to Login"):
            st.session_state['page'] = 'login'
            st.rerun()

def dashboard_page():
    st.sidebar.title(f"Welcome, {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()
    
    st.title("Audio Analysis Dashboard")
    st.write("Upload a music file to analyze its spectral content.")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Generate Spectrogram"):
            with st.spinner("Processing audio..."):
                try:
                    # Load audio (using librosa directly from the file object)
                    # Note: librosa.load can take a file-like object
                    y, sr = librosa.load(uploaded_file, sr=None)
                    
                    st.subheader("Mel Spectrogram")
                    fig = plot_mel_spectrogram(y, sr)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error processing audio: {e}")

# --- Main App Logic ---

def main():
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
