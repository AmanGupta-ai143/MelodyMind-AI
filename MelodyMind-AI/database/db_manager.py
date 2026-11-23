"""SQLite helper: insert / read / update / delete for all three tables."""
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

import config
from database.schema import SCHEMA


@contextmanager
def get_conn(db_path=None):
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True) if hasattr(path, "parent") else None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path=None) -> None:
    with get_conn(db_path) as c:
        c.executescript(SCHEMA)


# ---------------- generated_music ----------------
def add_composition(d: dict, db_path=None) -> int:
    with get_conn(db_path) as c:
        cur = c.execute(
            """INSERT INTO generated_music
               (composition_name, genre, mood, duration, midi_file_path, audio_file_path,
                model_version, num_notes, temperature, instrument)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (d["composition_name"], d.get("genre"), d.get("mood"), d.get("duration"), d["midi_file_path"],
             d.get("audio_file_path"), d.get("model_version"), d.get("num_notes", 0),
             d.get("temperature"), d.get("instrument")),
        )
        return cur.lastrowid


def get_composition(comp_id: int, db_path=None) -> Optional[dict]:
    with get_conn(db_path) as c:
        row = c.execute("SELECT * FROM generated_music WHERE id=?", (comp_id,)).fetchone()
        return dict(row) if row else None


def list_compositions(favorites_only: bool = False, limit: Optional[int] = None, db_path=None) -> List[dict]:
    q = ("SELECT g.*, (f.id IS NOT NULL) AS is_favorite FROM generated_music g "
         "LEFT JOIN favorites f ON f.composition_id = g.id ")
    if favorites_only:
        q += "WHERE f.id IS NOT NULL "
    q += "ORDER BY g.created_at DESC, g.id DESC"
    params: tuple = ()
    if limit:
        q += " LIMIT ?"
        params = (limit,)
    with get_conn(db_path) as c:
        return [dict(r) for r in c.execute(q, params).fetchall()]


def rename_composition(comp_id: int, name: str, db_path=None) -> None:
    with get_conn(db_path) as c:
        c.execute("UPDATE generated_music SET composition_name=? WHERE id=?", (name, comp_id))


def increment_download(comp_id: int, db_path=None) -> None:
    with get_conn(db_path) as c:
        c.execute("UPDATE generated_music SET download_count = download_count + 1 WHERE id=?", (comp_id,))


def delete_composition(comp_id: int, delete_files: bool = True, db_path=None) -> bool:
    comp = get_composition(comp_id, db_path)
    if not comp:
        return False
    with get_conn(db_path) as c:
        c.execute("DELETE FROM generated_music WHERE id=?", (comp_id,))
    if delete_files:
        for key in ("midi_file_path", "audio_file_path"):
            if comp.get(key):
                try:
                    config.abs_path(comp[key]).unlink(missing_ok=True)  # stays inside project
                except (ValueError, OSError):
                    pass
    return True


# ---------------- favorites ----------------
def is_favorite(comp_id: int, db_path=None) -> bool:
    with get_conn(db_path) as c:
        return c.execute("SELECT 1 FROM favorites WHERE composition_id=?", (comp_id,)).fetchone() is not None


def toggle_favorite(comp_id: int, db_path=None) -> bool:
    """Returns the new favourite state."""
    with get_conn(db_path) as c:
        row = c.execute("SELECT id FROM favorites WHERE composition_id=?", (comp_id,)).fetchone()
        if row:
            c.execute("DELETE FROM favorites WHERE id=?", (row["id"],))
            return False
        c.execute("INSERT INTO favorites (composition_id) VALUES (?)", (comp_id,))
        return True


# ---------------- training_history ----------------
def add_training_record(d: dict, db_path=None) -> int:
    with get_conn(db_path) as c:
        cur = c.execute(
            """INSERT INTO training_history
               (model_name, dataset_name, epochs, batch_size, learning_rate,
                final_training_loss, final_validation_loss, model_path)
               VALUES (?,?,?,?,?,?,?,?)""",
            (d["model_name"], d.get("dataset_name"), d.get("epochs"), d.get("batch_size"), d.get("learning_rate"),
             d.get("final_training_loss"), d.get("final_validation_loss"), d.get("model_path")),
        )
        return cur.lastrowid


def list_training_history(db_path=None) -> List[dict]:
    with get_conn(db_path) as c:
        return [dict(r) for r in c.execute("SELECT * FROM training_history ORDER BY id DESC").fetchall()]


# ---------------- aggregates for the dashboard ----------------
def summary(db_path=None) -> dict:
    with get_conn(db_path) as c:
        row = c.execute(
            "SELECT COUNT(*) n, COALESCE(AVG(duration),0) avg_d, COALESCE(SUM(num_notes),0) notes, "
            "COALESCE(SUM(download_count),0) dl FROM generated_music").fetchone()
        return {"total": row["n"], "avg_duration": row["avg_d"], "total_notes": row["notes"], "downloads": row["dl"]}


def genre_counts(db_path=None) -> dict:
    with get_conn(db_path) as c:
        rows = c.execute("SELECT genre, COUNT(*) n FROM generated_music WHERE genre IS NOT NULL AND genre NOT IN ('', 'None') "
                         "GROUP BY genre ORDER BY n DESC").fetchall()
        return {r["genre"]: r["n"] for r in rows}


def generation_activity(db_path=None) -> List[dict]:
    with get_conn(db_path) as c:
        rows = c.execute("SELECT date(created_at) d, COUNT(*) n FROM generated_music GROUP BY d ORDER BY d").fetchall()
        return [{"date": r["d"], "count": r["n"]} for r in rows]
