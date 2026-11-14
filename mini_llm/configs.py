"""
Пресеты обучения.

debug — прогон на ноутбуке (CPU) за пару минут: проверить, что код работает.
small — этап 0 на Kaggle (GPU): ~17M параметров, из них ~6M — эмбеддинги.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import GPTConfig


@dataclass
class TrainConfig:
    model: GPTConfig = field(default_factory=GPTConfig)

    # данные
    data_dir: str = "data/tokens"
    tokenizer: str = "artifacts/tokenizer/bpe_16k.json"

    # оптимизация
    batch_size: int = 64  # последовательностей в шаге
    grad_accum: int = 1  # накопление градиентов: эффективный batch = batch_size * grad_accum
    max_steps: int = 5000
    lr: float = 1e-3
    min_lr: float = 1e-4  # к концу косинусного расписания
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # оценка и логи
    eval_interval: int = 250
    eval_iters: int = 50  # батчей для оценки val loss
    log_interval: int = 20
    sample_prompt: str = "Москва — это"
    sample_tokens: int = 60

    # служебное
    out_dir: str = "checkpoints/stage0"
    seed: int = 1337
    compile: bool = False  # torch.compile: быстрее на GPU, но долгий первый шаг
    wandb: bool = False
    wandb_project: str = "my-mini-llm"
    run_name: str = "stage0-gpt2"


PRESETS: dict[str, TrainConfig] = {
    "debug": TrainConfig(
        model=GPTConfig(block_size=64, n_layer=2, n_head=2, n_embd=64),
        batch_size=16,
        max_steps=100,
        warmup_steps=10,
        eval_interval=50,
        eval_iters=5,
        log_interval=10,
        sample_tokens=20,
        out_dir="checkpoints/debug",
        run_name="debug",
    ),
    "small": TrainConfig(
        model=GPTConfig(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0),
    ),
}
