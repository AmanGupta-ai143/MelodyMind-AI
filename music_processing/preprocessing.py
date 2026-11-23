"""Dataset processing: MIDI -> melodies -> numeric tokens, plus dataset statistics.

Token = (pitch, duration_bucket). Each unique pair seen in the data gets an integer id.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

import config
from music_processing.midi_loader import EmptyDatasetError, iter_valid_midis, list_midi_files
from music_processing.note_extractor import extract_instruments, extract_melody, extract_notes


def duration_to_bucket(duration) -> np.ndarray:
    buckets = np.asarray(config.DURATION_BUCKETS)
    d = np.atleast_1d(np.asarray(duration, dtype=float))
    return np.abs(buckets[None, :] - d[:, None]).argmin(axis=1)


class Vocabulary:
    """Bidirectional mapping between (pitch, duration_bucket) tokens and integer ids."""

    def __init__(self, tokens: Sequence[Tuple[int, int]]):
        self.tokens: List[Tuple[int, int]] = [tuple(t) for t in tokens]
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}

    @classmethod
    def from_melodies(cls, melodies: Sequence[Sequence[Tuple[int, int]]]) -> "Vocabulary":
        return cls(sorted({tok for m in melodies for tok in m}))

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, melody: Sequence[Tuple[int, int]]) -> List[int]:
        return [self.token_to_id[t] for t in melody]

    def decode_id(self, idx: int) -> Tuple[int, float]:
        pitch, bucket = self.tokens[idx]
        return pitch, config.DURATION_BUCKETS[bucket]

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.tokens))

    @classmethod
    def load(cls, path) -> "Vocabulary":
        return cls([tuple(t) for t in json.loads(Path(path).read_text())])


def melody_to_tokens(melody: Sequence[Tuple[int, float]]) -> List[Tuple[int, int]]:
    if not melody:
        return []
    buckets = duration_to_bucket([d for _, d in melody])
    return [(int(p), int(b)) for (p, _), b in zip(melody, buckets)]


def process_dataset(
    dataset_dir=None,
    processed_dir=None,
    max_files: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    dataset_name: Optional[str] = None,
    min_melody_notes: int = 10,
) -> dict:
    """Load every MIDI file, extract melodies + real dataset statistics, save to data/processed."""
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = list_midi_files(dataset_dir, max_files=max_files)
    if not files:
        raise EmptyDatasetError("No MIDI files found in the dataset folder. Add .mid/.midi files first (see README).")

    pitch_counts = np.zeros(128, dtype=np.int64)
    dur_counts = np.zeros(len(config.DURATION_BUCKETS), dtype=np.int64)
    instrument_counts: dict = {}
    file_rows, skipped, raw_melodies = [], [], []
    total_notes = total_tracks = 0
    total_duration = 0.0

    for i, (path, pm, err) in enumerate(iter_valid_midis(files), start=1):
        if progress:
            progress(i, len(files))
        if err:
            skipped.append({"file": path.name, "reason": err})
            continue
        insts = extract_instruments(pm)
        for ins in insts:
            instrument_counts[ins["name"]] = instrument_counts.get(ins["name"], 0) + 1
        notes = extract_notes(pm)
        if len(notes):
            pitch_counts += np.bincount(notes["pitch"].to_numpy(), minlength=128)[:128]
            dur_counts += np.bincount(duration_to_bucket(notes["duration"].to_numpy()), minlength=len(dur_counts))
        melody = extract_melody(pm)
        if len(melody) >= min_melody_notes:
            raw_melodies.append(melody_to_tokens(melody))
        dur = float(pm.get_end_time())
        total_notes += len(notes)
        total_tracks += len(insts)
        total_duration += dur
        file_rows.append({
            "file": str(path.relative_to(dataset_dir)).replace("\\", "/"),
            "tracks": len(insts), "notes": int(len(notes)), "duration_s": round(dur, 1),
        })

    if not raw_melodies:
        raise EmptyDatasetError("No usable melodies could be extracted from the dataset.")

    vocab = Vocabulary.from_melodies(raw_melodies)
    encoded = [vocab.encode(m) for m in raw_melodies]

    stats = {
        "dataset_name": dataset_name or config.DATASET_NAME,
        "files_found": len(files),
        "files_processed": len(file_rows),
        "files_skipped": len(skipped),
        "skipped": skipped[:20],
        "total_tracks": total_tracks,
        "total_notes": int(total_notes),
        "total_duration_s": round(total_duration, 1),
        "melody_notes": int(sum(len(m) for m in encoded)),
        "vocab_size": len(vocab),
        "pitch_counts": pitch_counts.tolist(),
        "duration_labels": [f"{d}s" for d in config.DURATION_BUCKETS],
        "duration_counts": dur_counts.tolist(),
        "instrument_counts": instrument_counts,
        "files": file_rows[:1000],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (processed_dir / "melodies.json").write_text(json.dumps(encoded))
    vocab.save(processed_dir / "vocab.json")
    (processed_dir / "stats.json").write_text(json.dumps(stats))
    return stats


def load_processed(processed_dir=None) -> Tuple[List[List[int]], Vocabulary]:
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    mp, vp = processed_dir / "melodies.json", processed_dir / "vocab.json"
    if not mp.exists() or not vp.exists():
        raise EmptyDatasetError("Dataset has not been processed yet. Run the dataset processing step first.")
    return json.loads(mp.read_text()), Vocabulary.load(vp)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Process the MIDI dataset into training data.")
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()
    try:
        s = process_dataset(max_files=args.max_files, progress=lambda i, n: print(f"\r{i}/{n}", end="", file=sys.stderr))
    except EmptyDatasetError as e:
        sys.exit(f"\nError: {e}")
    print(f"\nProcessed {s['files_processed']} files ({s['files_skipped']} skipped); "
          f"{s['melody_notes']} melody notes; vocab size {s['vocab_size']}")
