"""LSTM next-note prediction model (PyTorch).

Input ids -> Embedding -> LSTM -> Dropout -> LSTM -> Dense (logits) -> Softmax (applied in
CrossEntropyLoss during training and explicitly with temperature during generation).
"""
import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        self.vocab_size, self.embed_dim, self.hidden_dim, self.dropout_p = vocab_size, embed_dim, hidden_dim, dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """x: (batch, time) ids -> logits (batch, time, vocab), new hidden state."""
        h1, h2 = (None, None) if hidden is None else hidden
        out = self.embedding(x)
        out, h1 = self.lstm1(out, h1)
        out = self.dropout(out)
        out, h2 = self.lstm2(out, h2)
        return self.fc(out), (h1, h2)

    def config(self) -> dict:
        return {"vocab_size": self.vocab_size, "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim, "dropout": self.dropout_p}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
