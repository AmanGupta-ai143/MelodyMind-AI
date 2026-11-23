"""Safe MIDI loading and validation."""
from pathlib import Path
from typing import Iterator, Optional

import pretty_midi

import config

MIDI_EXTENSIONS = {".mid", ".midi"}


class InvalidMidiError(Exception):
    """Raised when a file is not a usable MIDI file."""


class EmptyDatasetError(Exception):
    """Raised when the dataset folder has no (valid) MIDI files."""


def validate_midi_file(path, max_mb: float = config.MAX_MIDI_MB) -> pretty_midi.PrettyMIDI:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise InvalidMidiError(f"File not found: {p.name}")
    if p.suffix.lower() not in MIDI_EXTENSIONS:
        raise InvalidMidiError(f"Unsupported file format '{p.suffix}' (expected .mid or .midi): {p.name}")
    if p.stat().st_size > max_mb * 1024 * 1024:
        raise InvalidMidiError(f"File too large (> {max_mb} MB): {p.name}")
    if p.stat().st_size == 0:
        raise InvalidMidiError(f"Empty file: {p.name}")
    try:
        pm = pretty_midi.PrettyMIDI(str(p))
    except Exception as exc:  # mido raises many different error types
        raise InvalidMidiError(f"Could not parse MIDI file: {p.name}") from exc
    if not any(inst.notes for inst in pm.instruments):
        raise InvalidMidiError(f"MIDI file contains no notes: {p.name}")
    return pm


def list_midi_files(directory, max_files: Optional[int] = None, seed: int = 42):
    """Return sorted MIDI paths in a directory (recursively). Random subset if max_files."""
    import random

    d = Path(directory)
    if not d.exists():
        return []
    files = sorted(f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in MIDI_EXTENSIONS)
    if max_files and len(files) > max_files:
        files = sorted(random.Random(seed).sample(files, max_files))
    return files


def iter_valid_midis(files) -> Iterator[tuple]:
    """Yield (path, PrettyMIDI | None, error | None) so callers can log skipped files."""
    for f in files:
        try:
            yield f, validate_midi_file(f), None
        except InvalidMidiError as exc:
            yield f, None, str(exc)
