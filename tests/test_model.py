import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32)


def test_shapes_and_loss():
    torch.manual_seed(0)
    m = GPT(CFG)
    x = torch.randint(0, 100, (4, 16))
    logits, loss = m(x, x)
    assert logits.shape == (4, 16, 100)
    # до обучения loss ≈ ln(vocab_size): модель угадывает равномерно
    assert abs(loss.item() - torch.log(torch.tensor(100.0)).item()) < 0.5


def test_causal():
    """Изменение будущих токенов не должно влиять на предсказания для прошлых позиций."""
    torch.manual_seed(0)
    m = GPT(CFG).eval()
    x = torch.randint(0, 100, (1, 16))
    x2 = x.clone()
    x2[0, 10:] = (x2[0, 10:] + 1) % 100
    l1, _ = m(x, x)
    l2, _ = m(x2, x2)
    assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-5)
    assert not torch.allclose(l1[0, 10:], l2[0, 10:], atol=1e-5)


def test_weight_tying():
    m = GPT(CFG)
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


def test_overfits_one_batch():
    """Модель должна заучить один батч — базовая проверка, что обучение работает."""
    torch.manual_seed(0)
    m = GPT(CFG)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 100, (2, 16))
    y = torch.roll(x, -1, dims=1)
    for _ in range(200):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_length():
    m = GPT(CFG).eval()
    out = m.generate(torch.zeros((1, 3), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 23)  # контекст длиннее block_size обрезается внутри, но генерация продолжается
