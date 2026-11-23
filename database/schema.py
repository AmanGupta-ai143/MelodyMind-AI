"""SQLite schema: generated_music, training_history, favorites."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS generated_music (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    composition_name TEXT NOT NULL,
    genre TEXT,
    mood TEXT,
    duration REAL,
    midi_file_path TEXT NOT NULL,
    audio_file_path TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    num_notes INTEGER DEFAULT 0,
    temperature REAL,
    instrument TEXT,
    download_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS training_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    dataset_name TEXT,
    epochs INTEGER,
    batch_size INTEGER,
    learning_rate REAL,
    final_training_loss REAL,
    final_validation_loss REAL,
    training_date TEXT NOT NULL DEFAULT (datetime('now')),
    model_path TEXT
);

CREATE TABLE IF NOT EXISTS favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    composition_id INTEGER NOT NULL UNIQUE REFERENCES generated_music(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""
