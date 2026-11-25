from components.charts import plot
import pandas as pd
import streamlit as st

from components import charts, sidebar
from database import db_manager as db
from services import dataset_service as ds

sidebar.hero("Training Analytics", "Loss and evaluation curves from the actual training runs.")
h = ds.load_history()
plot(charts.training_loss_chart(h))
plot(charts.training_performance_chart(h))
st.caption("Music generation has no single 'accuracy' - loss and perplexity measure how well the model predicts the next note. "
           "Next-note accuracy is shown separately and is only a rough guide.")
if h:
    cfg = h["train_config"]
    st.markdown(f"**Latest run:** {len(h['epochs'])} epochs · batch {cfg['batch_size']} · seq len {cfg['seq_len']} · lr {cfg['learning_rate']:g} · "
                f"{h['train_sequences']:,} train / {h['val_sequences']:,} validation sequences")
if h:
    st.subheader("Per-epoch metrics (latest run)")
    st.dataframe(pd.DataFrame(h["epochs"]).rename(columns={
        "epoch": "Epoch", "train_loss": "Train loss", "val_loss": "Validation loss",
        "train_acc": "Train next-note acc", "val_acc": "Val next-note acc"}))
st.subheader("Training history (database)")
rows = db.list_training_history()
if rows:
    st.dataframe(pd.DataFrame(rows))
else:
    st.info("No training runs recorded yet. (On the free cloud app this table resets; see the per-epoch table above.)")
