"""Central configuration: paths and constants shared across the project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "midi_dataset"
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
MIDI_OUT = ROOT / "generated_music" / "midi"
AUDIO_OUT = ROOT / "generated_music" / "audio"
DB_PATH = ROOT / "data" / "melodymind.db"
LOGO_PATH = ROOT / "assets" / "logo.png"
CSS_PATH = ROOT / "assets" / "styles.css"

CHECKPOINT_PATH = CHECKPOINT_DIR / "melodymind_lstm.pt"
HISTORY_PATH = CHECKPOINT_DIR / "training_history.json"
STATS_PATH = PROCESSED_DIR / "stats.json"
MELODIES_PATH = PROCESSED_DIR / "melodies.json"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"

MODEL_NAME = "MelodyMind-LSTM"
MODEL_VERSION = "v1.0"
DATASET_NAME = os.environ.get("MELODYMIND_DATASET_NAME", "MAESTRO v3.0.0 (MIDI subset)")

MAX_MIDI_MB = 20          # reject larger files
MAX_AUDIO_SECONDS = 300   # cap for rendered audio
SAMPLE_RATE = 22050

# Note durations (seconds) are snapped to these buckets -> finite vocabulary
DURATION_BUCKETS = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0]

# General-MIDI programs the user can pick for playback (real, applied to the output MIDI)
INSTRUMENTS = {
    "Acoustic Grand Piano": 0,
    "Electric Piano": 4,
    "Harpsichord": 6,
    "Vibraphone": 11,
    "Nylon Guitar": 24,
    "Violin": 40,
    "Cello": 42,
    "Flute": 73,
    "Synth Lead": 80,
    "Warm Pad": 89,
}

# Tags saved with a composition. NOT used as model conditioning.
STYLE_LABELS = ["None", "Classical", "Jazz", "Pop", "Ambient", "Other"]
MOOD_LABELS = ["None", "Happy", "Sad", "Calm", "Energetic", "Peaceful"]


def ensure_dirs() -> None:
    for d in (DATASET_DIR, PROCESSED_DIR, CHECKPOINT_DIR, MIDI_OUT, AUDIO_OUT):
        d.mkdir(parents=True, exist_ok=True)


def abs_path(relative: str) -> Path:
    """Resolve a project-relative path and make sure it stays inside the project."""
    p = (ROOT / relative).resolve()
    if ROOT not in p.parents and p != ROOT:
        raise ValueError("Path outside project directory")
    return p


def rel_path(path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
