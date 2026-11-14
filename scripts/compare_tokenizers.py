"""
Сравнивает наш токенизатор с GPT-2, GPT-4, GPT-4o (tiktoken) и Qwen2.5
на валидационном русском тексте. Пишет отчёт в artifacts/tokenizer/comparison.md.

Пример:
    python scripts/compare_tokenizers.py --ours artifacts/tokenizer/bpe_16k.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.data import iter_documents  # noqa: E402
from mini_llm.tokenizer import BPETokenizer  # noqa: E402

DEMO_WORDS = [
    "привет",
    "здравствуйте",
    "токенизатор",
    "достопримечательности",
    "переосмысливающийся",
    "Москва",
    "1234567",
]
DEMO_SENTENCE = "Банк списал 1 499,90 ₽ за подписку «Яндекс Плюс» 12.03.2025."
COUNT_WORD = "пересмотреть"


@dataclass
class Tok:
    name: str
    vocab_size: int
    encode: Callable[[str], list[int]]
    pieces: Callable[[str], list[str]]  # для визуализации разбиения


def load_tokenizers(ours_paths: list[str]) -> list[Tok]:
    toks: list[Tok] = [
        Tok("символы (этап 0)", -1, lambda s: list(s), lambda s: list(s)),
        Tok("байты UTF-8", 256, lambda s: list(s.encode("utf-8")), lambda s: [f"<{b:02X}>" for b in s.encode()]),
    ]
    for p in ours_paths:
        bpe = BPETokenizer.load(p)
        toks.append(
            Tok(
                f"наш BPE ({Path(p).stem})",
                bpe.vocab_size,
                lambda s, b=bpe: b.encode(s, allowed_special=False),
                lambda s, b=bpe: [b.id_to_token(i) for i in b.encode(s, allowed_special=False)],
            )
        )

    try:
        import tiktoken

        for enc_name, label in [("gpt2", "GPT-2"), ("cl100k_base", "GPT-4"), ("o200k_base", "GPT-4o")]:
            enc = tiktoken.get_encoding(enc_name)

            def pieces(s: str, e=enc) -> list[str]:
                return [e.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in e.encode_ordinary(s)]

            toks.append(Tok(label, enc.n_vocab, lambda s, e=enc: e.encode_ordinary(s), pieces))
    except Exception as e:  # нет библиотеки или сети
        print(f"[warn] tiktoken недоступен: {e}")

    try:
        from tokenizers import Tokenizer

        qwen = Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        def qpieces(s: str, t=qwen) -> list[str]:
            return [t.decode([i]) for i in t.encode(s, add_special_tokens=False).ids]

        toks.append(
            Tok(
                "Qwen2.5",
                qwen.get_vocab_size(),
                lambda s, t=qwen: t.encode(s, add_special_tokens=False).ids,
                qpieces,
            )
        )
    except Exception as e:
        print(f"[warn] токенизатор Qwen недоступен: {e}")
    return toks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", default=["artifacts/tokenizer/bpe_16k.json"])
    ap.add_argument("--val", default="data/corpus/val.txt")
    ap.add_argument("--max-mb", type=float, default=5)
    ap.add_argument("--report", default="artifacts/tokenizer/comparison.md")
    args = ap.parse_args()

    docs = list(iter_documents(args.val, max_bytes=int(args.max_mb * 2**20)))
    n_chars = sum(len(d) for d in docs)
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    n_words = sum(len(re.findall(r"\p{L}+", d)) for d in docs)
    print(f"Валидация: {len(docs):,} документов, {n_chars:,} символов, {n_words:,} слов")

    toks = load_tokenizers(args.ours)
    lines: list[str] = []
    lines.append("# Сравнение токенизаторов на русском тексте\n")
    lines.append(f"Текст: `{args.val}`, {len(docs):,} документов, {n_chars:,} символов, {n_words:,} слов.\n")
    lines.append("| Токенизатор | Словарь | Токенов | Символов/токен ↑ | Токенов/слово ↓ | Длиннее нашего в |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    results = []
    for t in toks:
        n_tok = sum(len(t.encode(d)) for d in docs)
        results.append((t, n_tok))
        print(f"  {t.name:<22} {n_tok:>12,} токенов, {n_chars / n_tok:.2f} симв/ток")
    ours_n = next((n for t, n in results if t.name.startswith("наш")), None)
    for t, n_tok in results:
        vs = f"{n_tok / ours_n:.2f}×" if ours_n else "—"
        vocab = f"{t.vocab_size:,}" if t.vocab_size > 0 else "~100–200"
        lines.append(f"| {t.name} | {vocab} | {n_tok:,} | {n_chars / n_tok:.2f} | {n_tok / n_words:.2f} | {vs} |")

    lines.append("\n## Как режутся слова\n")
    for w in DEMO_WORDS + [DEMO_SENTENCE]:
        lines.append(f"**{w}**\n")
        for t in toks[2:]:
            p = t.pieces(w)
            lines.append(f"- {t.name} ({len(p)}): " + " · ".join(f"`{x}`" for x in p))
        lines.append("")

    lines.append("## Почему LLM плохо считают буквы\n")
    lines.append(
        f"Вопрос «сколько букв „р“ в слове „{COUNT_WORD}“?» модель видит не буквы, а токены. "
        "Буквы внутри токена для неё не видны напрямую — только id токена:\n"
    )
    for t in toks[2:]:
        p = t.pieces(COUNT_WORD)
        ids = t.encode(COUNT_WORD)
        lines.append(f"- {t.name}: " + " · ".join(f"`{x}`" for x in p) + f" → id {ids}")
    lines.append("")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nОтчёт: {report}")


if __name__ == "__main__":
    main()
