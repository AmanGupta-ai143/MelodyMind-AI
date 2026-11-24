from components.charts import plot
import streamlit as st

import config
from components import sidebar
from components.music_player import render_player
from components.piano_roll import piano_roll_figure
from database import db_manager as db
from models.generate import GenerationError, ModelLoadError
from music_processing.midi_converter import midi_file_to_notes
from music_processing.preprocessing import EmptyDatasetError
from services import dataset_service as ds
from services.generation_service import GenerationParams, generate_composition

sidebar.hero("AI Music Generator", "Every note below is predicted by the trained LSTM - nothing is picked from a MIDI folder.")

if not ds.model_is_trained():
    st.warning("No trained model yet. Go to **Model Training Center**, process the dataset and train first.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.subheader("Controls")
    instrument = st.selectbox("Instrument (playback sound)", list(config.INSTRUMENTS))
    num_notes = st.slider("Number of notes", 16, 512, 128, step=16, help="Sequence length. Duration follows from the notes generated.")
    temperature = st.slider("Temperature (randomness)", 0.3, 1.8, 1.0, 0.05,
                            help="Low = safe and repetitive, high = adventurous but messier.")
    seed_in = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)
    with st.expander("Tags (saved with the composition)"):
        genre = st.selectbox("Style label", config.STYLE_LABELS)
        mood = st.selectbox("Mood label", config.MOOD_LABELS)
        st.caption("These are metadata tags only. The LSTM is not conditioned on genre or mood.")
    gen_clicked = st.button("🎼 Generate", type="primary")
    regen_clicked = st.button("🔄 Regenerate (new seed)")

if gen_clicked or regen_clicked:
    params = GenerationParams(num_notes=num_notes, temperature=temperature, instrument=instrument,
                              seed=None if (regen_clicked or seed_in == 0) else int(seed_in),
                              genre_label=genre, mood_label=mood)
    try:
        with st.spinner("The LSTM is composing..."):
            st.session_state["last_result"] = generate_composition(params)
    except (ModelLoadError, GenerationError, EmptyDatasetError) as exc:
        st.session_state.pop("last_result", None)   # never show music from a failed run
        st.error(str(exc))
    except Exception:
        st.session_state.pop("last_result", None)
        st.error("Unexpected error while generating music.")

res = st.session_state.get("last_result")
with right:
    if not res:
        st.info("Choose your settings and press **Generate**.")
    else:
        st.subheader(res["composition_name"])
        st.caption(f"{res['num_notes']} notes · {res['duration']:.1f}s · {res['instrument']} · temperature {res['temperature']} · "
                   f"seed {res['seed']} · {res['model_version']}")
        if res.get("warning"):
            st.warning(res["warning"])
        try:
            plot(piano_roll_figure(midi_file_to_notes(config.abs_path(res["midi_file_path"])), "Generated piano roll"))
        except Exception:
            st.caption("Piano roll unavailable.")
        render_player(res, key="gen")
        if res.get("id"):
            st.success("Saved to your library.")
        elif st.button("💾 Save to library"):
            res["id"] = db.add_composition({k: res[k] for k in (
                "composition_name", "genre", "mood", "duration", "midi_file_path", "audio_file_path",
                "model_version", "num_notes", "temperature", "instrument")})
            st.session_state["last_result"] = res
            st.rerun()
