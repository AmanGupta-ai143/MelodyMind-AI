import torch

from models.lstm_model import MusicLSTM


def test_model_init_and_forward_shape():
    m = MusicLSTM(vocab_size=50, embed_dim=8, hidden_dim=16)
    logits, hidden = m(torch.randint(0, 50, (4, 10)))
    assert logits.shape == (4, 10, 50) and len(hidden) == 2
    assert m.num_parameters() > 0


def test_one_training_step_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    m = MusicLSTM(20, 8, 32, dropout=0.0)
    x, y = torch.randint(0, 20, (32, 8)), torch.randint(0, 20, (32,))
    opt, loss_fn = torch.optim.Adam(m.parameters(), lr=1e-2), torch.nn.CrossEntropyLoss()
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(m(x)[0][:, -1], y)
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first
