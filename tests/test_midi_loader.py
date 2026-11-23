import pytest

from music_processing.midi_loader import EmptyDatasetError, InvalidMidiError, list_midi_files, validate_midi_file
from music_processing.note_extractor import extract_instruments, extract_melody, extract_notes
from tests.helpers import make_midi


def test_valid_midi_loads(tmp_path):
    pm = validate_midi_file(make_midi(tmp_path / "a.mid"))
    assert len(pm.instruments[0].notes) == 60


def test_invalid_midi_rejected(tmp_path):
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"this is not midi")
    with pytest.raises(InvalidMidiError):
        validate_midi_file(bad)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "song.mp3"
    f.write_bytes(b"x")
    with pytest.raises(InvalidMidiError):
        validate_midi_file(f)


def test_file_size_limit(tmp_path):
    p = make_midi(tmp_path / "big.mid")
    with pytest.raises(InvalidMidiError):
        validate_midi_file(p, max_mb=0.00001)


def test_list_files_and_empty_dir(tmp_path):
    assert list_midi_files(tmp_path) == []
    make_midi(tmp_path / "a.mid")
    assert len(list_midi_files(tmp_path)) == 1


def test_note_and_instrument_extraction(tmp_path):
    pm = validate_midi_file(make_midi(tmp_path / "a.mid", program=40))
    assert extract_instruments(pm)[0]["name"] == "Violin"
    df = extract_notes(pm)
    assert list(df.columns[:4]) == ["pitch", "start", "end", "duration"] and len(df) == 60
    melody = extract_melody(pm)
    assert len(melody) == 60 and all(isinstance(p, int) for p, _ in melody)
