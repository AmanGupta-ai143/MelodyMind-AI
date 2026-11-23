"""Extract instruments, notes and a monophonic melody from a PrettyMIDI object."""
from typing import List, Tuple

import pandas as pd
import pretty_midi


def instrument_name(inst: pretty_midi.Instrument) -> str:
    if inst.is_drum:
        return "Drums"
    return pretty_midi.program_to_instrument_name(inst.program)


def extract_instruments(pm: pretty_midi.PrettyMIDI) -> List[dict]:
    return [
        {"name": instrument_name(i), "program": int(i.program), "is_drum": bool(i.is_drum), "num_notes": len(i.notes)}
        for i in pm.instruments
        if i.notes
    ]


def extract_notes(pm: pretty_midi.PrettyMIDI, include_drums: bool = False) -> pd.DataFrame:
    """All notes as a DataFrame: pitch, start, end, duration, velocity, instrument."""
    rows = []
    for inst in pm.instruments:
        if inst.is_drum and not include_drums:
            continue
        name = instrument_name(inst)
        for n in inst.notes:
            rows.append((n.pitch, n.start, n.end, n.end - n.start, n.velocity, name))
    df = pd.DataFrame(rows, columns=["pitch", "start", "end", "duration", "velocity", "instrument"])
    return df.sort_values(["start", "pitch"]).reset_index(drop=True)


def extract_melody(pm: pretty_midi.PrettyMIDI, min_duration: float = 0.05, group_window: float = 0.03) -> List[Tuple[int, float]]:
    """Approximate 'skyline' monophonic melody -> [(pitch, duration_seconds), ...].

    All non-drum notes are merged and sorted by onset (highest pitch first on ties).
    Notes starting within `group_window` seconds of the previous kept note are dropped,
    so each onset keeps a single note. This gives a manageable monophonic representation.
    """
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, -n.pitch))
    melody, last_start = [], None
    for n in notes:
        if last_start is not None and n.start - last_start < group_window:
            continue
        dur = n.end - n.start
        if dur < min_duration:
            continue
        melody.append((int(n.pitch), float(dur)))
        last_start = n.start
    return melody
