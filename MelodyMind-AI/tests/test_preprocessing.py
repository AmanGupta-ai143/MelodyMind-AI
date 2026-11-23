import numpy as np
import pytest

from music_processing.midi_loader import EmptyDatasetError
from music_processing.preprocessing import Vocabulary, duration_to_bucket, melody_to_tokens, process_dataset
from music_processing.sequence_generator import make_sequences, split_melodies
from tests.helpers import make_midi


def test_duration_bucket():
    assert duration_to_bucket(0.5)[0] == 3
    assert duration_to_bucket(10)[0] == 7


def test_vocab_roundtrip(tmp_path):
    toks = melody_to_tokens([(60, 0.5), (62, 0.25), (60, 0.5)])
    v = Vocabulary.from_melodies([toks])
    ids = v.encode(toks)
    assert len(v) == 2 and ids[0] == ids[2]
    assert v.decode_id(ids[1]) == (62, 0.25)
    v.save(tmp_path / "v.json")
    assert Vocabulary.load(tmp_path / "v.json").tokens == v.tokens


def test_make_sequences_next_note():
    X, y = make_sequences([[1, 2, 3, 4, 5, 6]], seq_len=3)
    assert X.shape == (3, 3) and y.tolist() == [4, 5, 6] and X[0].tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        make_sequences([[1, 2]], seq_len=5)


def test_split_has_no_overlap_and_handles_single_piece():
    tr, va = split_melodies([[1] * 5, [2] * 5, [3] * 5, [4] * 5], 0.25)
    assert len(tr) == 3 and len(va) == 1
    tr, va = split_melodies([list(range(100))], 0.2)
    assert len(tr[0]) == 80 and len(va[0]) == 20


def test_process_dataset(tmp_path):
    ds, out = tmp_path / "ds", tmp_path / "out"
    ds.mkdir()
    make_midi(ds / "a.mid")
    make_midi(ds / "b.mid", program=40, base=55)
    (ds / "broken.mid").write_bytes(b"junk")
    s = process_dataset(ds, out)
    assert s["files_processed"] == 2 and s["files_skipped"] == 1
    assert sum(s["pitch_counts"]) == 120 and (out / "melodies.json").exists()


def test_empty_dataset(tmp_path):
    with pytest.raises(EmptyDatasetError):
        process_dataset(tmp_path / "nothing", tmp_path / "out")
