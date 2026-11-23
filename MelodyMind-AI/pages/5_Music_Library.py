import streamlit as st

from components import sidebar
from components.music_player import render_player
from database import db_manager as db

sidebar.hero("Music Library", "Your saved compositions.")
only_fav = st.toggle("Favorites only")
items = db.list_compositions(favorites_only=only_fav)
if not items:
    st.info("No compositions here yet. Generate one and press **Save to library**.")
for c in items:
    star = "⭐ " if c["is_favorite"] else ""
    with st.expander(f"{star}{c['composition_name']} · {c['duration']:.0f}s · {c['created_at']}"):
        st.caption(f"Style tag: {c['genre'] or '-'} · Mood tag: {c['mood'] or '-'} · Instrument: {c['instrument'] or '-'} · "
                   f"Model: {c['model_version']} · Notes: {c['num_notes']} · Temperature: {c['temperature']}")
        render_player(c, key=f"lib{c['id']}")
        a, b = st.columns(2)
        if a.button("Remove favorite" if c["is_favorite"] else "☆ Favorite", key=f"fav{c['id']}"):
            db.toggle_favorite(c["id"])
            st.rerun()
        if b.button("🗑 Delete", key=f"del{c['id']}"):
            db.delete_composition(c["id"])
            st.rerun()
