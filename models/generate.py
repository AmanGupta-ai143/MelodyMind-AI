"""Load a trained checkpoint and generate note sequences with temperature sampling."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch

import config
from models.lstm_model import MusicLSTM
from music_processing.preprocessing import Vocabulary


class ModelLoadError(Exception):
    pass


class GenerationError(Exception):
    pass


@dataclass
class LoadedModel:
    model: MusicLSTM
    vocab: Vocabulary
    meta: dict


def load_checkpoint(path=None) -> LoadedModel:
    path = Path(path or config.CHECKPOINT_PATH)
    if not path.exists():
        raise ModelLoadError("No trained model found. Train the model in the Model Training Center first.")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelLoadError("Checkpoint file is corrupted or unreadable.") from exc
    missing = {"state_dict", "model_config", "vocab"} - set(ckpt)
    if missing:
        raise ModelLoadError(f"Invalid checkpoint (missing: {', '.join(sorted(missing))}).")
    vocab = Vocabulary([tuple(t) for t in ckpt["vocab"]])
    cfg = ckpt["model_config"]
    if cfg.get("vocab_size") != len(vocab):
        raise ModelLoadError("Checkpoint vocabulary size does not match the model configuration.")
    try:
        model = MusicLSTM(**cfg)
        model.load_state_dict(ckpt["state_dict"])
    except Exception as exc:
        raise ModelLoadError("Checkpoint weights do not match the model architecture.") from exc
    model.eval()
    meta = {k: ckpt.get(k) for k in ("version", "epoch", "val_loss", "train_config", "saved_at")}
    return LoadedModel(model, vocab, meta)


def sampling_probs(logits: torch.Tensor, temperature: float, top_k: int = 0) -> torch.Tensor:
    """Softmax with temperature (low = conservative, high = adventurous), optional top-k."""
    temperature = max(float(temperature), 0.05)
    scaled = logits.float() / temperature
    if top_k and top_k < scaled.numel():
        kth = torch.topk(scaled, top_k).values[-1]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    return torch.softmax(scaled, dim=-1)


def generate_tokens(model: MusicLSTM, seed_ids: Sequence[int], length: int, temperature: float = 1.0,
                    top_k: int = 0, rng_seed: Optional[int] = None) -> List[int]:
    """Autoregressively predict `length` new token ids after the seed sequence."""
    if length < 1:
        raise GenerationError("Length must be at least 1.")
    if len(seed_ids) == 0:
        raise GenerationError("Seed sequence is empty.")
    gen = torch.Generator().manual_seed(int(rng_seed)) if rng_seed is not None else None
    model.eval()
    out: List[int] = []
    with torch.no_grad():
        logits, hidden = model(torch.tensor([list(seed_ids)], dtype=torch.long))
        last = logits[0, -1]
        for _ in range(length):
            probs = sampling_probs(last, temperature, top_k)
            if not torch.isfinite(probs).all():
                raise GenerationError("Model produced invalid probabilities.")
            nxt = int(torch.multinomial(probs, 1, generator=gen).item())
            out.append(nxt)
            logits, hidden = model(torch.tensor([[nxt]], dtype=torch.long), hidden)
            last = logits[0, -1]
    return out
