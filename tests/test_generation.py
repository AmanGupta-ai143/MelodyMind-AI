import pytest
import torch

import config
from database import db_manager as db
from models.generate import GenerationError, ModelLoadError, generate_tokens, load_checkpoint, sampling_probs
from models.lstm_model import MusicLSTM
from music_processing.midi_converter import tokens_to_midi
from music_processing.preprocessing import Vocabulary
from services.audio_service import render_wav


def _vocab():
    return Vocabulary([(60 + i, 3) for i in range(8)])


def test_temperature_changes_distribution():
    logits = torch.tensor([2.0, 1.0, 0.0])
    low, high = sampling_probs(logits, 0.2), sampling_probs(logits, 2.0)
    assert low.max() > high.max() and abs(low.sum().item() - 1) < 1e-5


def test_generate_tokens_valid_and_reproducible():
    m = MusicLSTM(8, 8, 16)
    a = generate_tokens(m, [0, 1, 2], 20, 1.0, rng_seed=5)
    b = generate_tokens(m, [0, 1, 2], 20, 1.0, rng_seed=5)
    assert a == b and len(a) == 20 and all(0 <= t < 8 for t in a)
    with pytest.raises(GenerationError):
        generate_tokens(m, [], 5)


def test_checkpoint_roundtrip_and_validation(tmp_path):
    v, m = _vocab(), MusicLSTM(8, 8, 16)
    ck = tmp_path / "m.pt"
    torch.save({"state_dict": m.state_dict(), "model_config": m.config(), "vocab": [list(t) for t in v.tokens],
                "version": "t", "train_config": {"seq_len": 4}}, ck)
    loaded = load_checkpoint(ck)
    assert len(loaded.vocab) == 8 and loaded.meta["train_config"]["seq_len"] == 4
    with pytest.raises(ModelLoadError):
        load_checkpoint(tmp_path / "missing.pt")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"garbage")
    with pytest.raises(ModelLoadError):
        load_checkpoint(bad)
    torch.save({"state_dict": {}}, tmp_path / "partial.pt")
    with pytest.raises(ModelLoadError):
        load_checkpoint(tmp_path / "partial.pt")


def test_midi_conversion_and_audio(tmp_path):
    pm = tokens_to_midi([0, 1, 2, 3], _vocab(), program=40)
    assert len(pm.instruments[0].notes) == 4 and pm.instruments[0].program == 40
    path = tmp_path / "x.mid"
    pm.write(str(path))
    method = render_wav(path, tmp_path / "x.wav")
    assert (tmp_path / "x.wav").stat().st_size > 1000 and method in {"numpy-synth", "fluidsynth"}
    with pytest.raises(ValueError):
        tokens_to_midi([], _vocab())


def test_database_crud(tmp_path):
    dbp = tmp_path / "t.db"
    db.init_db(dbp)
    cid = db.add_composition({"composition_name": "a", "genre": "Jazz", "duration": 10.0, "midi_file_path": "generated_music/midi/a.mid",
                              "num_notes": 5}, dbp)
    assert db.get_composition(cid, dbp)["composition_name"] == "a"
    assert db.toggle_favorite(cid, dbp) is True and db.is_favorite(cid, dbp)
    assert len(db.list_compositions(favorites_only=True, db_path=dbp)) == 1
    db.increment_download(cid, dbp)
    s = db.summary(dbp)
    assert s["total"] == 1 and s["downloads"] == 1 and s["total_notes"] == 5
    assert db.genre_counts(dbp) == {"Jazz": 1}
    db.add_training_record({"model_name": "m", "epochs": 1}, dbp)
    assert len(db.list_training_history(dbp)) == 1
    assert db.delete_composition(cid, delete_files=False, db_path=dbp) is True
    assert db.get_composition(cid, dbp) is None and not db.is_favorite(cid, dbp)
