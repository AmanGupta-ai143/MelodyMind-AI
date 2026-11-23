"""MelodyMind AI - entry point.  Run:  streamlit run app.py"""
import streamlit as st

import config
from components import sidebar
from database import db_manager as db

st.set_page_config(page_title="MelodyMind AI", page_icon="🎵", layout="wide")
config.ensure_dirs()
db.init_db()
sidebar.apply_theme()
sidebar.render_sidebar_header()

pages = [
    st.Page("pages/1_Dashboard.py", title="Dashboard", icon="🏠", default=True),
    st.Page("pages/2_Music_Generator.py", title="AI Music Generator", icon="🎼"),
    st.Page("pages/3_Dataset_Explorer.py", title="Dataset Explorer", icon="🗂️"),
    st.Page("pages/4_Model_Training.py", title="Model Training Center", icon="🧠"),
    st.Page("pages/5_Music_Library.py", title="Music Library", icon="📚"),
    st.Page("pages/6_Playback_Studio.py", title="Playback Studio", icon="🎧"),
    st.Page("pages/7_Training_Analytics.py", title="Training Analytics", icon="📈"),
    st.Page("pages/8_About.py", title="About Project", icon="ℹ️"),
]
st.navigation(pages).run()
