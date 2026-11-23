"""Convert generated token ids to MIDI, and MIDI back to a note table."""
from pathlib import Path
from typing import Sequence

import pandas as pd
import pretty_midi

from music_processing.note_extractor import extract_notes
from music_processing.preprocessing import Vocabulary


def tokens_to_midi(token_ids: Sequence[int], vocab: Vocabulary, program: int = 0, velocity: int = 85, gap: float = 0.0) -> pretty_midi.PrettyMIDI:
    if len(token_ids) == 0:
        raise ValueError("Cannot create MIDI from an empty sequence.")
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=int(program))
    t = 0.0
    for tid in token_ids:
        pitch, dur = vocab.decode_id(int(tid))
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=int(pitch), start=t, end=t + dur))
        t += dur + gap
    pm.instruments.append(inst)
    return pm


def save_midi(pm: pretty_midi.PrettyMIDI, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return path


def midi_file_to_notes(path) -> pd.DataFrame:
    return extract_notes(pretty_midi.PrettyMIDI(str(path)))
