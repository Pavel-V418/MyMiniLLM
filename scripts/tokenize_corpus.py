"""
Токенизирует data/corpus/{train,val}.txt в data/tokens/{train,val}.bin.

Формат .bin: подряд идущие id токенов в uint16 (2 байта на токен; хватает,
пока vocab_size < 65536). После каждого документа ставится <|endoftext|>,
чтобы модель училась понимать, где кончается один текст и начинается другой.

Кодирование распараллелено на все ядра процессора.

    python scripts/tokenize_corpus.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.data import iter_documents  # noqa: E402
from mini_llm.tokenizer import BPETokenizer  # noqa: E402

_tok: BPETokenizer | None = None
_eot: int = -1


def _init_worker(tokenizer_path: str) -> None:
    # каждый процесс загружает свою копию токенизатора (и свой кэш)
    global _tok, _eot
    _tok = BPETokenizer.load(tokenizer_path)
    _eot = _tok.special_tokens["<|endoftext|>"]


def _encode_doc(text: str) -> np.ndarray:
    ids = _tok.encode(text, allowed_special=False)  # спецтокены в сыром тексте — просто текст
    ids.append(_eot)
    return np.asarray(ids, dtype=np.uint16)


def tokenize_file(src: Path, dst: Path, tokenizer_path: str, workers: int) -> int:
    t0 = time.time()
    n_tokens = 0
    with open(dst, "wb") as f, Pool(workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
        docs = iter_documents(src)
        for k, arr in enumerate(pool.imap(_encode_doc, docs, chunksize=16)):
            arr.tofile(f)
            n_tokens += len(arr)
            if k % 2000 == 0 and k:
                print(f"  {src.name}: {k:,} документов, {n_tokens:,} токенов, {time.time() - t0:.0f} с")
    dt = time.time() - t0
    print(f"{src.name} -> {dst.name}: {n_tokens:,} токенов за {dt:.0f} с")
    return n_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="data/tokens")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/bpe_16k.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    assert tok.vocab_size < 2**16, "uint16 не вместит такой словарь"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": args.tokenizer, "vocab_size": tok.vocab_size, "dtype": "uint16"}
    for split in ("train", "val"):
        meta[f"{split}_tokens"] = tokenize_file(
            Path(args.corpus) / f"{split}.txt", out / f"{split}.bin", args.tokenizer, args.workers
        )
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {meta}")


if __name__ == "__main__":
    main()
