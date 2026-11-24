from components.charts import plot
import streamlit as st

import config
from components import sidebar
from components.music_player import render_player
from components.piano_roll import piano_roll_figure
from database import db_manager as db
from music_processing.midi_converter import midi_file_to_notes

sidebar.hero("Playback Studio", "Listen, inspect and download a saved composition.")
items = db.list_compositions()
if not items:
    st.info("No saved compositions yet.")
    st.stop()
c = st.selectbox("Composition", items, format_func=lambda r: f"{r['composition_name']} ({r['duration']:.0f}s)")
st.caption(f"{c['num_notes']} notes · {c['instrument']} · temperature {c['temperature']} · {c['model_version']} · {c['created_at']}")
render_player(c, key=f"studio{c['id']}")
try:
    plot(piano_roll_figure(midi_file_to_notes(config.abs_path(c["midi_file_path"])), "Piano roll"))
except Exception:
    st.warning("The MIDI file for this composition could not be read.")
