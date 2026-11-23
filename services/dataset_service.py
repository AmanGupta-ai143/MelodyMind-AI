"""Read dataset statistics and training history produced by the real pipeline."""
import json
from pathlib import Path
from typing import List, Optional

import config
from music_processing.midi_loader import MIDI_EXTENSIONS


def get_stats() -> Optional[dict]:
    if not config.STATS_PATH.exists():
        return None
    try:
        return json.loads(config.STATS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_history() -> Optional[dict]:
    if not config.HISTORY_PATH.exists():
        return None
    try:
        h = json.loads(config.HISTORY_PATH.read_text())
        return h if h.get("epochs") else None
    except (OSError, json.JSONDecodeError):
        return None


def model_is_trained() -> bool:
    return config.CHECKPOINT_PATH.exists() and load_history() is not None


def dataset_file_count() -> int:
    if not config.DATASET_DIR.exists():
        return 0
    return sum(1 for f in config.DATASET_DIR.rglob("*") if f.suffix.lower() in MIDI_EXTENSIONS)


def resolve_dataset_file(relative: str) -> Path:
    """Safely map a listed dataset file name to a path inside the dataset folder."""
    p = (config.DATASET_DIR / relative).resolve()
    if config.DATASET_DIR.resolve() not in p.parents:
        raise ValueError("Invalid dataset file")
    return p
