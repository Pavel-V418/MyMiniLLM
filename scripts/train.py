"""
Обучение GPT (этап 0).

На ноутбуке (проверка кода, ~1–2 минуты на CPU):
    python scripts/train.py --config debug

На Kaggle (GPU):
    python scripts/train.py --config small --wandb
    python scripts/train.py --config small --resume          # продолжить с последнего чекпоинта

Любой параметр пресета можно переопределить: --max-steps 2000 --lr 6e-4 --batch-size 32
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.configs import PRESETS, TrainConfig  # noqa: E402
from mini_llm.model import GPT, GPTConfig  # noqa: E402
from mini_llm.tokenizer import BPETokenizer  # noqa: E402


# ------------------------------------------------------------------ данные
class TokenData:
    """Читает .bin через memmap: файл не грузится в память целиком."""

    def __init__(self, data_dir: str, block_size: int, batch_size: int, device: torch.device):
        self.splits = {s: np.memmap(Path(data_dir) / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        for s, arr in self.splits.items():
            if len(arr) <= block_size + 1:
                raise ValueError(f"{s}.bin слишком мал: {len(arr)} токенов")

    def batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Случайные отрезки длиной block_size. y — это x, сдвинутый на 1 токен вперёд."""
        data = self.splits[split]
        ix = np.random.randint(0, len(data) - self.block_size - 1, size=self.batch_size)
        x = torch.from_numpy(np.stack([data[i : i + self.block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + self.block_size] for i in ix]).astype(np.int64))
        if self.device.type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


# -------------------------------------------------------- learning rate
def lr_at(step: int, cfg: TrainConfig) -> float:
    """Линейный warmup, затем косинусное затухание от lr до min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


# ------------------------------------------------------------- конфиг
def build_config() -> tuple[TrainConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="debug", choices=list(PRESETS))
    ap.add_argument("--resume", action="store_true", help="продолжить с out_dir/last.pt")
    ap.add_argument("--device", default=None, help="cuda / cpu (по умолчанию — автоматически)")
    # переопределения пресета: --max-steps 1000 и т.п.
    for f in dataclasses.fields(TrainConfig):
        if f.name == "model":
            continue
        t = type(f.default)
        if t is bool:
            ap.add_argument(f"--{f.name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=None)
        else:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=t, default=None)
    for f in dataclasses.fields(GPTConfig):
        if f.name == "vocab_size":
            continue
        t = type(f.default)
        if t is bool:
            ap.add_argument(f"--{f.name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=None)
        else:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=t, default=None)
    args = ap.parse_args()

    base = PRESETS[args.config]
    train_over = {f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)
                  if f.name != "model" and getattr(args, f.name) is not None}
    model_over = {f.name: getattr(args, f.name) for f in dataclasses.fields(GPTConfig)
                  if f.name != "vocab_size" and getattr(args, f.name) is not None}
    cfg = dataclasses.replace(base, model=dataclasses.replace(base.model, **model_over), **train_over)
    return cfg, args


# -------------------------------------------------------------- main
def main() -> None:
    cfg, args = build_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Mixed precision: на GPU считаем в 16 битах (в 2–3 раза быстрее), на CPU — в fp32.
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast = torch.autocast("cuda", dtype=amp_dtype)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        amp_dtype = torch.float32
        autocast = nullcontext()
    # fp16 легко переполняется -> GradScaler масштабирует loss. Для bf16/fp32 не нужен.
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    tok = BPETokenizer.load(cfg.tokenizer)
    cfg.model.vocab_size = tok.vocab_size
    eot = tok.special_tokens["<|endoftext|>"]

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GPT(cfg.model).to(device)
    optimizer = model.configure_optimizer(cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2), device.type)

    step = 0
    best_val = float("inf")
    if args.resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        best_val = ckpt.get("best_val", best_val)
        print(f"Продолжаем с шага {step} (best val {best_val:.3f})")

    raw_model = model  # для сохранения: у скомпилированной модели другие имена весов
    if cfg.compile:
        model = torch.compile(model)

    data = TokenData(cfg.data_dir, cfg.model.block_size, cfg.batch_size, device)
    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.model.block_size

    print(f"Устройство: {device}, точность: {amp_dtype}")
    print(f"Параметров: {raw_model.num_params() / 1e6:.2f}M "
          f"(без эмбеддингов {raw_model.num_params(non_embedding=True) / 1e6:.2f}M)")
    print(f"Данные: train {len(data.splits['train']):,} токенов, val {len(data.splits['val']):,}")
    print(f"Токенов за шаг: {tokens_per_step:,}, всего за обучение: {tokens_per_step * cfg.max_steps / 1e6:.1f}M "
          f"(~{tokens_per_step * cfg.max_steps / len(data.splits['train']):.1f} эпох)")

    wandb_run = None
    if cfg.wandb:
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=dataclasses.asdict(cfg),
                               resume="allow")

    metrics_file = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")

    def log(row: dict) -> None:
        metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log(row, step=row["step"])

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                x, y = data.batch(split)
                with autocast:
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    @torch.no_grad()
    def sample() -> str:
        model.eval()
        idx = torch.tensor([tok.encode(cfg.sample_prompt)], dtype=torch.long, device=device)
        with autocast:
            out = raw_model.generate(idx, cfg.sample_tokens, temperature=0.8, top_k=50, eos_id=eot)
        model.train()
        return tok.decode(out[0].tolist())

    def save(name: str) -> None:
        torch.save(
            {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best_val": best_val,
                "model_config": dataclasses.asdict(cfg.model),
                "train_config": dataclasses.asdict(cfg),
            },
            out_dir / name,
        )

    # ---------------------------------------------------- цикл обучения
    model.train()
    x, y = data.batch("train")
    t0 = time.time()
    while step < cfg.max_steps:
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # оценка, пример генерации, чекпоинт
        if step % cfg.eval_interval == 0:
            losses = evaluate()
            ppl = math.exp(losses["val"])
            print(f"\n[eval] step {step}: train {losses['train']:.3f}, val {losses['val']:.3f}, perplexity {ppl:.1f}")
            text = sample()
            print(f"[sample] {text!r}\n")
            log({"step": step, "train/loss_eval": losses["train"], "val/loss": losses["val"], "val/ppl": ppl})
            if wandb_run is not None:
                import wandb

                wandb_run.log({"sample": wandb.Html(f"<pre>{text}</pre>")}, step=step)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save("best.pt")
            save("last.pt")

        # шаг оптимизации (с накоплением градиентов)
        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            with autocast:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            x, y = data.batch("train")  # следующий батч грузим, пока GPU считает backward
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        else:
            grad_norm = float("nan")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * cfg.log_interval / dt
            print(f"step {step:>6}/{cfg.max_steps} | loss {loss_acc:.3f} | lr {lr:.2e} | "
                  f"grad {grad_norm:.2f} | {tps:,.0f} ток/с")
            log({"step": step, "train/loss": loss_acc, "lr": lr, "grad_norm": grad_norm, "tokens_per_sec": tps})

    losses = evaluate()
    print(f"\nГотово. val loss {losses['val']:.3f}, perplexity {math.exp(losses['val']):.1f}, best {best_val:.3f}")
    log({"step": step, "train/loss_eval": losses["train"], "val/loss": losses["val"],
         "val/ppl": math.exp(losses["val"])})
    if losses["val"] < best_val:
        best_val = losses["val"]
        save("best.pt")
    save("last.pt")
    print(f"Чекпоинты: {out_dir / 'best.pt'}, {out_dir / 'last.pt'}")
    metrics_file.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
