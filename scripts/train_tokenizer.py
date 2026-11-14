"""
Обучает BPE-токенизатор на data/corpus/train.txt.

Пример:
    python scripts/train_tokenizer.py --vocab-size 16000 --max-mb 50
Результат: artifacts/tokenizer/bpe_16k.json (+ .vocab.txt для просмотра)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.data import iter_documents  # noqa: E402
from mini_llm.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus/train.txt")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--max-mb", type=float, default=50, help="сколько МБ корпуса использовать для обучения")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default=None, help="путь без расширения; по умолчанию artifacts/tokenizer/bpe_<N>k")
    args = ap.parse_args()

    out = Path(args.out or f"artifacts/tokenizer/bpe_{args.vocab_size // 1000}k")
    max_bytes = int(args.max_mb * 2**20) if args.max_mb else None

    t0 = time.time()
    tok = BPETokenizer.train(
        iter_documents(args.input, max_bytes=max_bytes),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"Обучение заняло {time.time() - t0:.0f} с, размер словаря: {tok.vocab_size}")

    tok.save(out)
    print(f"Сохранено: {out.with_suffix('.json')}, {out.with_suffix('.vocab.txt')}")

    sample = "Привет! Это мой собственный токенизатор для русского языка."
    ids = tok.encode(sample)
    print(f"\n{sample!r}\n-> {len(ids)} токенов: {[tok.id_to_token(i) for i in ids]}")
    assert tok.decode(ids) == sample


if __name__ == "__main__":
    main()
