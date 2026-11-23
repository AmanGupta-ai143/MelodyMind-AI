import plotly.graph_objects as go

from components.charts import _style, empty_figure, note_name


def piano_roll_figure(notes_df, title: str = "Piano roll", max_notes: int = 3000) -> go.Figure:
    """Time on X, MIDI pitch on Y. One line segment per note."""
    if notes_df is None or len(notes_df) == 0:
        return empty_figure(title, "No notes to display.")
    df = notes_df.head(max_notes)
    xs, ys = [], []
    for s, e, p in zip(df["start"], df["end"], df["pitch"]):
        xs += [s, e, None]
        ys += [p, p, None]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(width=5, color="#7c5cff"), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=df["start"], y=df["pitch"], mode="markers", marker=dict(size=4, color="#2fa8ff"), name="Note onset",
                    customdata=[note_name(int(p)) for p in df["pitch"]],
                    hovertemplate="t=%{x:.2f}s<br>pitch %{y} (%{customdata})<extra></extra>")
    fig.update_layout(hovermode="closest")
    return _style(fig, title, "Time (seconds)", "MIDI pitch", height=380)
