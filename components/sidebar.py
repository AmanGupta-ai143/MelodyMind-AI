"""Global page chrome: CSS, brand header, light/dark switch."""
import streamlit as st

import config

LIGHT_CSS = """
<style>
.stApp { background:#f4f5fb; color:#14173a; }
section[data-testid="stSidebar"] { background:#ffffff; }
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp span,.stApp li { color:#14173a; }
.mm-card { background:#ffffff; border-color:#dfe2f5; } .mm-card .value { color:#14173a; }
.mm-card .label,.mm-card .sub { color:#5b6190; }
header[data-testid="stHeader"] { background:#f4f5fb; }
[data-testid="stMarkdownContainer"] { color:#14173a; }
</style>
"""


def current_theme() -> str:
    return "light" if st.session_state.get("light_theme") else "dark"


def apply_theme() -> None:
    if config.CSS_PATH.exists():
        st.markdown(f"<style>{config.CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
    if current_theme() == "light":
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def render_sidebar_header() -> None:
    with st.sidebar:
        if config.LOGO_PATH.exists():
            st.image(str(config.LOGO_PATH), width=64)
        st.markdown("### MelodyMind AI")
        st.caption("Intelligent Music Composition Studio")
        st.toggle("Light theme", key="light_theme")
        st.divider()


def hero(title: str, subtitle: str) -> None:
    # inline colours so the banner text is readable in both light and dark mode
    st.markdown(
        f'<div class="mm-hero"><div style="font-size:2rem;font-weight:700;color:#ffffff;line-height:1.2;">{title}</div>'
        f'<div style="margin-top:.35rem;color:#dfe2ff;">{subtitle}</div></div>',
        unsafe_allow_html=True)
