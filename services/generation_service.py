"""High-level composition generation: trained LSTM -> tokens -> MIDI -> WAV."""
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import config
from models.generate import GenerationError, ModelLoadError, generate_tokens, load_checkpoint
from music_processing.midi_converter import save_midi, tokens_to_midi
from music_processing.preprocessing import load_processed
from services.audio_service import AudioRenderError, render_wav

_CACHE: dict = {}


@dataclass
class GenerationParams:
    num_notes: int = 128
    temperature: float = 1.0
    instrument: str = "Acoustic Grand Piano"
    seed: Optional[int] = None
    genre_label: str = "None"   # metadata only - the model is NOT conditioned on it
    mood_label: str = "None"    # metadata only


def get_model():
    """Load the checkpoint once and reuse it (re-load when the file changes). Never retrains."""
    key = config.CHECKPOINT_PATH.stat().st_mtime if config.CHECKPOINT_PATH.exists() else None
    if _CACHE.get("key") != key or key is None:
        _CACHE["model"] = load_checkpoint()
        _CACHE["key"] = key
    return _CACHE["model"]


def generate_composition(p: GenerationParams) -> dict:
    if not 1 <= p.num_notes <= 2000:
        raise GenerationError("Number of notes must be between 1 and 2000.")
    if p.instrument not in config.INSTRUMENTS:
        raise GenerationError("Unknown instrument.")
    loaded = get_model()  # may raise ModelLoadError
    seq_len = int((loaded.meta.get("train_config") or {}).get("seq_len", 32))

    melodies, _ = load_processed()
    seeds = [m for m in melodies if len(m) > seq_len]
    if not seeds:
        raise GenerationError("No seed melody available. Re-process the dataset.")
    seed = p.seed if p.seed is not None else random.randrange(1, 10**6)
    rng = random.Random(seed)
    src = rng.choice(seeds)
    start = rng.randrange(0, len(src) - seq_len)
    seed_ids = src[start:start + seq_len]

    ids = generate_tokens(loaded.model, seed_ids, p.num_notes, p.temperature, top_k=0, rng_seed=seed)
    pm = tokens_to_midi(ids, loaded.vocab, program=config.INSTRUMENTS[p.instrument])
    if not any(i.notes for i in pm.instruments):
        raise GenerationError("Generation produced no notes.")

    config.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"MelodyMind_{stamp}_{seed}"
    midi_path = save_midi(pm, config.MIDI_OUT / f"{name}.mid")
    audio_rel, audio_method, warning = None, None, None
    try:
        wav = config.AUDIO_OUT / f"{name}.wav"
        audio_method = render_wav(midi_path, wav)
        audio_rel = config.rel_path(wav)
    except AudioRenderError as exc:
        warning = f"MIDI created, but audio rendering failed: {exc}"

    return {
        "id": None, "composition_name": name, "genre": p.genre_label, "mood": p.mood_label,
        "duration": float(pm.get_end_time()), "midi_file_path": config.rel_path(midi_path),
        "audio_file_path": audio_rel, "model_version": f"{config.MODEL_NAME} {loaded.meta.get('version') or ''}".strip(),
        "num_notes": len(ids), "temperature": p.temperature, "instrument": p.instrument,
        "seed": seed, "audio_method": audio_method, "warning": warning,
    }
