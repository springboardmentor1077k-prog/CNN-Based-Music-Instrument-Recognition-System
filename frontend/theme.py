import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
        body {
            background-color: #0f172a;
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #6366f1;
            color: white;
            border-radius: 8px;
            height: 3em;
        }
        .stFileUploader {
            border: 2px dashed #6366f1;
            border-radius: 10px;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
