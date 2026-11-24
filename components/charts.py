"""Plotly charts. Every chart is built from real data; no data -> a clear empty state."""
import math
from typing import List, Optional

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PURPLE, BLUE, PINK = "#7c5cff", "#2fa8ff", "#ff6fb1"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(p: int) -> str:
    return f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"


def _template() -> str:
    return "plotly_white" if st.session_state.get("light_theme") else "plotly_dark"


def _style(fig: go.Figure, title: str, x: str = "", y: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(template=_template(), title=title, xaxis_title=x, yaxis_title=y, height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      margin=dict(l=40, r=20, t=55, b=40), legend=dict(orientation="h", y=-0.2))
    return fig


def empty_figure(title: str, message: str, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14), xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title, height=height)


def training_loss_chart(history: Optional[dict]) -> go.Figure:
    title = "Model training loss"
    if not history or not history.get("epochs"):
        return empty_figure(title, "No training run yet. Train the model in the Model Training Center.")
    ep = [e["epoch"] for e in history["epochs"]]
    fig = go.Figure()
    fig.add_scatter(x=ep, y=[e["train_loss"] for e in history["epochs"]], name="Training loss", mode="lines+markers", line=dict(color=PURPLE))
    fig.add_scatter(x=ep, y=[e["val_loss"] for e in history["epochs"]], name="Validation loss", mode="lines+markers", line=dict(color=BLUE))
    return _style(fig, title, "Epoch", "Cross-entropy loss")


def training_performance_chart(history: Optional[dict]) -> go.Figure:
    title = "Training performance"
    if not history or not history.get("epochs"):
        return empty_figure(title, "No training run yet.")
    E = history["epochs"]
    ep = [e["epoch"] for e in E]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Perplexity (exp of loss)", "Next-note prediction accuracy (top-1)"))
    fig.add_scatter(x=ep, y=[math.exp(e["train_loss"]) for e in E], name="Train perplexity", line=dict(color=PURPLE), row=1, col=1)
    fig.add_scatter(x=ep, y=[math.exp(e["val_loss"]) for e in E], name="Val perplexity", line=dict(color=BLUE), row=1, col=1)
    fig.add_scatter(x=ep, y=[e["train_acc"] for e in E], name="Train next-note acc.", line=dict(color=PURPLE, dash="dot"), row=1, col=2)
    fig.add_scatter(x=ep, y=[e["val_acc"] for e in E], name="Val next-note acc.", line=dict(color=BLUE, dash="dot"), row=1, col=2)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Perplexity", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", row=1, col=2)
    return _style(fig, title)


def pitch_histogram(pitch_counts: Optional[List[int]]) -> go.Figure:
    title = "MIDI note distribution (training dataset)"
    if not pitch_counts or sum(pitch_counts) == 0:
        return empty_figure(title, "Dataset not processed yet.")
    pitches = list(range(128))
    fig = go.Figure(go.Bar(x=pitches, y=pitch_counts, marker_color=PURPLE,
                           customdata=[note_name(p) for p in pitches],
                           hovertemplate="Pitch %{x} (%{customdata})<br>Count %{y}<extra></extra>"))
    nz = [p for p, c in enumerate(pitch_counts) if c]
    fig.update_xaxes(range=[min(nz) - 2, max(nz) + 2])
    return _style(fig, title, "MIDI pitch", "Frequency")


def instrument_donut(counts: Optional[dict], top: int = 8) -> go.Figure:
    title = "Instrument distribution (tracks)"
    if not counts:
        return empty_figure(title, "Dataset not processed yet.")
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    head, tail = items[:top], items[top:]
    if tail:
        head.append(("Other", sum(v for _, v in tail)))
    fig = go.Figure(go.Pie(labels=[k for k, _ in head], values=[v for _, v in head], hole=0.55))
    return _style(fig, title)


def duration_distribution(labels, counts) -> go.Figure:
    title = "Note duration distribution"
    if not counts or sum(counts) == 0:
        return empty_figure(title, "Dataset not processed yet.")
    fig = go.Figure(go.Bar(x=labels, y=counts, marker_color=BLUE))
    return _style(fig, title, "Note duration (nearest bucket)", "Number of notes")


def generation_activity_chart(rows: List[dict]) -> go.Figure:
    title = "Music generation activity"
    if not rows:
        return empty_figure(title, "No saved compositions yet.")
    fig = go.Figure(go.Scatter(x=[r["date"] for r in rows], y=[r["count"] for r in rows], mode="lines+markers", line=dict(color=PINK)))
    return _style(fig, title, "Date", "Compositions saved")


def plot(fig, key=None):
    """Render a Plotly figure with our own template (theme=None stops Streamlit overriding colours)."""
    st.plotly_chart(fig, theme=None, key=key)
