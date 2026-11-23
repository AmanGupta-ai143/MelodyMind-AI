"""Turn encoded melodies into (input sequence -> next note) training pairs."""
from typing import List, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def split_melodies(melodies: Sequence[Sequence[int]], val_fraction: float = 0.1, seed: int = 42):
    """Split by piece (no leakage). With a single piece, split it chronologically."""
    melodies = [list(m) for m in melodies]
    if len(melodies) >= 2:
        idx = np.random.RandomState(seed).permutation(len(melodies))
        n_val = max(1, int(round(len(melodies) * val_fraction)))
        val = [melodies[i] for i in idx[:n_val]]
        train = [melodies[i] for i in idx[n_val:]]
        return train, val
    m = melodies[0]
    cut = int(len(m) * (1 - val_fraction))
    return [m[:cut]], [m[cut:]]


def make_sequences(melodies: Sequence[Sequence[int]], seq_len: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """X: (N, seq_len) previous notes, y: (N,) the note that follows. Example: C4 D4 E4 G4 -> A4."""
    xs, ys = [], []
    for m in melodies:
        arr = np.asarray(m, dtype=np.int16)
        if len(arr) <= seq_len:
            continue
        windows = sliding_window_view(arr, seq_len + 1)[::stride]
        xs.append(windows[:, :-1])
        ys.append(windows[:, -1])
    if not xs:
        raise ValueError(f"No melody is longer than the sequence length ({seq_len}).")
    return np.concatenate(xs), np.concatenate(ys)
