"""Train the LSTM on the processed dataset. Run: python -m models.train --epochs 30"""
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from models.lstm_model import MusicLSTM
from music_processing.preprocessing import load_processed
from music_processing.sequence_generator import make_sequences, split_melodies


class TrainingError(Exception):
    pass


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    seq_len: int = 32
    learning_rate: float = 1e-3
    embed_dim: int = 64
    hidden_dim: int = 256
    dropout: float = 0.3
    val_fraction: float = 0.1
    max_sequences: int = 300_000
    seed: int = 42


def _evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = correct = n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)[0][:, -1]
            total_loss += loss_fn(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def train_model(cfg: TrainConfig, progress_callback: Optional[Callable[[dict, int], None]] = None,
                save_to_db: bool = True) -> dict:
    """Real training loop. Returns the history dict. Saves best-validation checkpoint."""
    try:
        melodies, vocab = load_processed()
        torch.manual_seed(cfg.seed)
        rng = np.random.RandomState(cfg.seed)

        train_m, val_m = split_melodies(melodies, cfg.val_fraction, cfg.seed)
        xtr, ytr = make_sequences(train_m, cfg.seq_len)
        xva, yva = make_sequences(val_m, cfg.seq_len)
        if len(xtr) > cfg.max_sequences:
            sel = rng.choice(len(xtr), cfg.max_sequences, replace=False)
            xtr, ytr = xtr[sel], ytr[sel]
        if len(xva) > cfg.max_sequences // 5:
            sel = rng.choice(len(xva), cfg.max_sequences // 5, replace=False)
            xva, yva = xva[sel], yva[sel]

        def loader(x, y, shuffle):
            ds = TensorDataset(torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64)))
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)

        tr_loader, va_loader = loader(xtr, ytr, True), loader(xva, yva, False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MusicLSTM(len(vocab), cfg.embed_dim, cfg.hidden_dim, cfg.dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        history = {"epochs": [], "train_config": asdict(cfg), "train_sequences": int(len(xtr)),
                   "val_sequences": int(len(xva)), "device": str(device), "vocab_size": len(vocab),
                   "num_parameters": model.num_parameters()}
        best_val = math.inf
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            run_loss = correct = n = 0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)[0][:, -1]
                loss = loss_fn(logits, yb)
                if not torch.isfinite(loss):
                    raise TrainingError("Loss became NaN/inf - try a lower learning rate.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                n += len(yb)
            val_loss, val_acc = _evaluate(model, va_loader, loss_fn, device)
            entry = {"epoch": epoch, "train_loss": run_loss / n, "val_loss": val_loss,
                     "train_acc": correct / n, "val_acc": val_acc}
            history["epochs"].append(entry)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "model_config": model.config(),
                    "vocab": [list(t) for t in vocab.tokens],
                    "train_config": asdict(cfg),
                    "epoch": epoch, "val_loss": val_loss,
                    "version": config.MODEL_VERSION,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                }, config.CHECKPOINT_PATH)
                history["best_epoch"] = epoch
                history["best_val_loss"] = val_loss
            config.HISTORY_PATH.write_text(json.dumps(history))
            if progress_callback:
                progress_callback(entry, cfg.epochs)

        if save_to_db:
            from database import db_manager as db
            db.init_db()
            from services import dataset_service
            last = history["epochs"][-1]
            stats = dataset_service.get_stats() or {}
            db.add_training_record({
                "model_name": config.MODEL_NAME, "dataset_name": stats.get("dataset_name", config.DATASET_NAME),
                "epochs": cfg.epochs, "batch_size": cfg.batch_size, "learning_rate": cfg.learning_rate,
                "final_training_loss": last["train_loss"], "final_validation_loss": last["val_loss"],
                "model_path": config.rel_path(config.CHECKPOINT_PATH),
            })
        return history
    except TrainingError:
        raise
    except Exception as exc:
        raise TrainingError(f"Training failed: {exc}") from exc


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    for f, v in asdict(TrainConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    a = vars(ap.parse_args())
    cfg = TrainConfig(**a)
    t0 = time.time()
    try:
        train_model(cfg, lambda e, total: print(
            f"epoch {e['epoch']}/{total}  train_loss={e['train_loss']:.4f}  val_loss={e['val_loss']:.4f}  "
            f"val_next_note_acc={e['val_acc']:.3f}"))
    except Exception as e:
        sys.exit(f"Error: {e}")
    print(f"Done in {time.time() - t0:.0f}s. Checkpoint: {config.rel_path(config.CHECKPOINT_PATH)}")
