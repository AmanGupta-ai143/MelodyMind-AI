import streamlit as st

import config
from components import charts, sidebar
from music_processing.midi_loader import EmptyDatasetError
from music_processing.preprocessing import process_dataset
from services import dataset_service as ds

sidebar.hero("Model Training Center", "Process the MIDI dataset, then train the LSTM. Weights are saved and reused for generation.")

st.subheader("1 · Dataset")
n_files = ds.dataset_file_count()
st.write(f"MIDI files in dataset folder: **{n_files}**")
if n_files == 0:
    st.warning("The dataset folder is empty. Run `python scripts/download_dataset.py` (MAESTRO subset) "
               "or copy your own .mid files into `data/midi_dataset/`.")
max_files = st.number_input("Max files to process (0 = all)", 0, 5000, 150, step=10)
if st.button("⚙️ Process dataset", disabled=n_files == 0):
    bar = st.progress(0.0, text="Reading MIDI files...")
    try:
        s = process_dataset(max_files=int(max_files) or None, progress=lambda i, n: bar.progress(i / n, text=f"{i}/{n} files"))
        st.success(f"Processed {s['files_processed']} files, {s['melody_notes']:,} melody notes, vocabulary {s['vocab_size']}.")
    except EmptyDatasetError as exc:
        st.error(str(exc))
stats = ds.get_stats()
if stats:
    st.caption(f"Ready: {stats['files_processed']} files · {stats['melody_notes']:,} melody notes · vocab {stats['vocab_size']}")

st.subheader("2 · Architecture")
st.code("Input sequence → Embedding → LSTM → Dropout → LSTM → Dense → Softmax → next note", language="text")
st.caption("Token = (pitch, duration bucket). The model predicts the next token from the previous N tokens.")

st.subheader("3 · Training")
c1, c2, c3, c4 = st.columns(4)
epochs = c1.number_input("Epochs", 1, 200, 20)
batch = c2.selectbox("Batch size", [32, 64, 128, 256], index=2)
seq_len = c3.selectbox("Sequence length", [16, 32, 48, 64], index=1)
lr = c4.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3, format_func=lambda v: f"{v:g}")
d1, d2 = st.columns(2)
hidden = d1.selectbox("LSTM hidden units", [128, 256, 512], index=1)
dropout = d2.slider("Dropout", 0.0, 0.6, 0.3, 0.05)

if st.button("🚀 Start training", type="primary", disabled=stats is None):
    from models.train import TrainConfig, TrainingError, train_model
    bar = st.progress(0.0, text="Starting...")
    chart_box = st.empty()
    live = {"epochs": []}

    def on_epoch(entry, total):
        live["epochs"].append(entry)
        bar.progress(entry["epoch"] / total, text=f"Epoch {entry['epoch']}/{total} · train {entry['train_loss']:.3f} · val {entry['val_loss']:.3f}")
        chart_box.plotly_chart(charts.training_loss_chart(live), key=f"live_{entry['epoch']}")

    try:
        train_model(TrainConfig(epochs=int(epochs), batch_size=int(batch), seq_len=int(seq_len), learning_rate=float(lr),
                                hidden_dim=int(hidden), dropout=float(dropout)), on_epoch)
        st.success("Training finished. Best checkpoint saved.")
    except (TrainingError, EmptyDatasetError) as exc:
        st.error(str(exc))

st.subheader("Status")
h = ds.load_history()
if h and config.CHECKPOINT_PATH.exists():
    st.markdown(f"Checkpoint saved · best epoch **{h.get('best_epoch')}** · best validation loss **{h.get('best_val_loss', float('nan')):.3f}** · "
                f"{h['num_parameters']:,} parameters · trained on {h['device']}")
else:
    st.info("No trained checkpoint yet.")
