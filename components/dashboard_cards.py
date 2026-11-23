import streamlit as st


def stat_card(icon: str, label: str, value, sub: str = "") -> str:
    return (f'<div class="mm-card"><div class="icon">{icon}</div><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def stat_row(cards) -> None:
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        col.markdown(stat_card(*c), unsafe_allow_html=True)


def status_pill(text: str, ok: bool) -> str:
    return f'<span class="mm-pill {"mm-ok" if ok else "mm-warn"}">{text}</span>'
