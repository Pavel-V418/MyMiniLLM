"""
Скачивает кусок русского корпуса с Hugging Face (streaming, без загрузки
всего датасета) и сохраняет в data/corpus/{train,val}.txt.

Документы разделяются строкой <|endoftext|>.

Примеры:
    python scripts/prepare_corpus.py --mb 100
    python scripts/prepare_corpus.py --dataset HuggingFaceFW/fineweb-2 --config rus_Cyrl --mb 200
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DOC_SEP = "\n<|endoftext|>\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikimedia/wikipedia")
    ap.add_argument("--config", default="20231101.ru")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--mb", type=float, default=100, help="сколько мегабайт текста набрать")
    ap.add_argument("--val-frac", type=float, default=0.02, help="доля документов в валидацию")
    ap.add_argument("--min-chars", type=int, default=200, help="выкинуть совсем короткие документы")
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset  # импорт здесь, чтобы --help работал без библиотеки

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    budget = int(args.mb * 1024 * 1024)

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    written = 0
    n_train = n_val = 0
    with open(out / "train.txt", "w", encoding="utf-8") as ftr, open(out / "val.txt", "w", encoding="utf-8") as fva:
        for ex in ds:
            text = (ex.get(args.text_field) or "").strip()
            if len(text) < args.min_chars:
                continue
            if rng.random() < args.val_frac:
                fva.write(text + DOC_SEP)
                n_val += 1
            else:
                ftr.write(text + DOC_SEP)
                n_train += 1
            written += len(text.encode("utf-8"))
            if (n_train + n_val) % 2000 == 0:
                print(f"  {written / 2**20:7.1f} / {args.mb:.0f} MB, документов: {n_train + n_val:,}")
            if written >= budget:
                break

    print(f"Готово: train={n_train:,} документов, val={n_val:,}, {written / 2**20:.1f} MB -> {out}/")


if __name__ == "__main__":
    main()
