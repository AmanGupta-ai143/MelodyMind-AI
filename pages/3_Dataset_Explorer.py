from components.charts import plot
import pandas as pd
import streamlit as st

import config
from components import charts, sidebar
from components.piano_roll import piano_roll_figure
from music_processing.midi_converter import midi_file_to_notes
from music_processing.midi_loader import InvalidMidiError, validate_midi_file
from services import dataset_service as ds
from services.audio_service import AudioRenderError, render_wav

sidebar.hero("Dataset Explorer", "Understand exactly what the model was trained on.")
stats = ds.get_stats()

if not stats:
    n = ds.dataset_file_count()
    st.info(f"{n} MIDI file(s) found in the dataset folder, but the dataset has not been processed yet. "
            "Use the **Model Training Center** (or `python -m music_processing.preprocessing`).")
    st.stop()

from components.dashboard_cards import stat_row
stat_row([
    ("🎹", "MIDI files used", stats["files_processed"], f"{stats['files_skipped']} skipped"),
    ("🎻", "Tracks", stats["total_tracks"], "instrument tracks"),
    ("🎵", "Notes", f"{stats['total_notes']:,}", f"{stats['melody_notes']:,} in melodies"),
    ("⏱️", "Total duration", f"{stats['total_duration_s'] / 3600:.1f} h", f"vocab size {stats['vocab_size']}"),
])
st.write("")
a, b = st.columns(2)
a.plotly_chart(charts.pitch_histogram(stats["pitch_counts"]), theme=None, config=charts.NO_BAR)
b.plotly_chart(charts.instrument_donut(stats["instrument_counts"]), theme=None, config=charts.NO_BAR)
plot(charts.duration_distribution(stats["duration_labels"], stats["duration_counts"]))

st.subheader("Dataset files")
df = pd.DataFrame(stats["files"])
st.dataframe(df, height=260)
if stats["skipped"]:
    with st.expander(f"Skipped files ({stats['files_skipped']})"):
        st.dataframe(pd.DataFrame(stats["skipped"]))

st.subheader("Preview a file")


def _exists(name: str) -> bool:
    try:
        return ds.resolve_dataset_file(name).is_file()
    except ValueError:
        return False


available = [f for f in df["file"].tolist() if _exists(f)]
if not available:
    st.info("File preview is unavailable here because the MIDI files are not stored in the repository (licence/size). "
            "Run the project locally with the dataset to preview files.")
else:
    st.caption(f"{len(available)} of {len(df)} dataset files are available for preview on this server.")
    choice = st.selectbox("File", available)
    if st.button("Preview (first 30 s)"):
        try:
            path = ds.resolve_dataset_file(choice)
            validate_midi_file(path)
            notes = midi_file_to_notes(path)
            plot(piano_roll_figure(notes[notes["start"] < 30], f"Piano roll - {choice}"))
            wav = config.AUDIO_OUT / "_preview.wav"
            render_wav(path, wav, max_seconds=30)
            st.audio(wav.read_bytes(), format="audio/wav")
        except (InvalidMidiError, AudioRenderError, ValueError) as exc:
            st.error(str(exc))

st.subheader("Source, license and attribution")
st.markdown(f"""
**Dataset:** {stats['dataset_name']}.  
The default dataset is the **MAESTRO v3.0.0** MIDI files released by the Magenta team at Google, licensed
**CC BY-NC-SA 4.0** (non-commercial, attribution, share-alike): <https://magenta.tensorflow.org/datasets/maestro>.
If you trained on your own MIDI files, make sure you have the right to use them and update this section.
Genre metadata is not shown because the dataset has no reliable genre labels (it is classical piano).
""")
