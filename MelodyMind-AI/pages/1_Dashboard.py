import streamlit as st

from components import charts, dashboard_cards as cards, sidebar
from database import db_manager as db
from services import dataset_service as ds

sidebar.hero("MelodyMind AI", "Welcome! Train an LSTM on real MIDI data, then compose new music with it.")
summary, stats, history = db.summary(), ds.get_stats(), ds.load_history()
trained = ds.model_is_trained()

epochs_done = len(history["epochs"]) if history else 0
best_val = history.get("best_val_loss") if history else None
cards.stat_row([
    ("🎵", "Compositions generated", summary["total"], "saved to library"),
    ("🎹", "MIDI files processed", stats["files_processed"] if stats else 0, "training dataset"),
    ("🔁", "Training epochs completed", epochs_done, "latest run"),
    ("🎼", "Generated notes", summary["total_notes"], "across saved compositions"),
])
st.write("")
c1, c2, c3 = st.columns([1, 1, 1])
c1.markdown("**Model status** &nbsp;" + cards.status_pill("Trained" if trained else "Not trained", trained), unsafe_allow_html=True)
if best_val is not None:
    last = history["epochs"][-1]
    c2.markdown(f"**Best validation loss:** {best_val:.3f}  \n**Val next-note accuracy (top-1):** {last['val_acc']:.1%}")
else:
    c2.caption("No evaluation metrics yet.")
if c3.button("⚡ Quick generate", type="primary"):
    st.switch_page("pages/2_Music_Generator.py")

st.plotly_chart(charts.training_loss_chart(history))
a, b = st.columns(2)
a.plotly_chart(charts.pitch_histogram(stats["pitch_counts"] if stats else None))
b.plotly_chart(charts.instrument_donut(stats["instrument_counts"] if stats else None))

st.subheader("Recently generated compositions")
recent = db.list_compositions(limit=5)
if not recent:
    st.info("Nothing saved yet - generate a composition and press **Save to library**.")
for r in recent:
    st.markdown(f"**{r['composition_name']}** · {r['duration']:.0f}s · {r['num_notes']} notes · {r['created_at']}")

x, y = st.columns(2)
x.plotly_chart(charts.generation_activity_chart(db.generation_activity()))
gc = db.genre_counts()
with y:
    st.markdown("**Most used style labels** (tags only, not model conditioning)")
    st.bar_chart(gc) if gc else st.caption("No labelled compositions yet.")
