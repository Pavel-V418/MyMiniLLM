"""
Генерация текста обученной моделью.

    python scripts/generate.py --ckpt checkpoints/stage0/best.pt --prompt "Москва — это"
    python scripts/generate.py --ckpt checkpoints/stage0/best.pt           # интерактивно
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.model import GPT, GPTConfig  # noqa: E402
from mini_llm.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/stage0/best.pt")
    ap.add_argument("--tokenizer", default=None, help="по умолчанию — тот, что записан в чекпоинте")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer or ckpt["train_config"]["tokenizer"])
    eot = tok.special_tokens["<|endoftext|>"]
    print(f"Модель: {model.num_params() / 1e6:.1f}M параметров, шаг {ckpt['step']}, val loss {ckpt['best_val']:.3f}")

    def run(prompt: str) -> None:
        idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        for _ in range(args.num_samples):
            out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, eos_id=eot)
            print("\n" + tok.decode(out[0].tolist()).replace("<|endoftext|>", " [конец]") + "\n" + "-" * 60)

    if args.prompt is not None:
        run(args.prompt)
        return
    print("Введите начало текста (пустая строка — выход)")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        run(line)


if __name__ == "__main__":
    main()
