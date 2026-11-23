import streamlit as st

import config
from database import db_manager as db


def render_player(comp: dict, key: str) -> None:
    """Audio player (play/pause/seek built in) + download buttons. Works for saved or unsaved compositions."""
    comp_id = comp.get("id")
    audio_path = None
    if comp.get("audio_file_path"):
        try:
            audio_path = config.abs_path(comp["audio_file_path"])
        except ValueError:
            audio_path = None
    if audio_path and audio_path.exists():
        st.audio(audio_path.read_bytes(), format="audio/wav")
    else:
        st.info("No rendered audio for this composition - download the MIDI file instead.")

    c1, c2 = st.columns(2)
    try:
        midi_path = config.abs_path(comp["midi_file_path"])
        if midi_path.exists():
            c1.download_button("⬇ Download MIDI", midi_path.read_bytes(), file_name=midi_path.name, mime="audio/midi",
                               key=f"dl_mid_{key}",
                               on_click=db.increment_download if comp_id else None, args=(comp_id,) if comp_id else None)
    except ValueError:
        pass
    if audio_path and audio_path.exists():
        c2.download_button("⬇ Download WAV", audio_path.read_bytes(), file_name=audio_path.name, mime="audio/wav",
                           key=f"dl_wav_{key}")
