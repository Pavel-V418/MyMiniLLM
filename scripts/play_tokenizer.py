"""
Песочница: вводишь текст — видишь, как токенизатор его режет.

    python scripts/play_tokenizer.py                 # интерактивный режим
    python scripts/play_tokenizer.py "Привет, мир!"  # один текст
    python scripts/play_tokenizer.py --explain "пересмотреть"   # пошагово: какие склейки применились

В интерактивном режиме:
    текст          -> токены
    ?слово         -> пошаговое объяснение склеек
    #123           -> что за токен с id 123
    пустая строка  -> выход
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.tokenizer import BPETokenizer  # noqa: E402

COLORS = ["\033[30;46m", "\033[30;43m", "\033[30;42m", "\033[30;45m", "\033[30;47m"]
RESET = "\033[0m"


def show(tok: BPETokenizer, text: str) -> None:
    ids = tok.encode(text)
    pieces = [tok.id_to_token(i) for i in ids]
    colored = "".join(COLORS[k % len(COLORS)] + p.replace(" ", "·") + RESET for k, p in enumerate(pieces))
    n_chars, n_bytes = len(text), len(text.encode("utf-8"))
    print(f"\n  {colored}")
    print(f"  токенов: {len(ids)} | символов: {n_chars} | байт: {n_bytes} | сжатие: {n_chars / max(len(ids), 1):.2f} симв/ток")
    print(f"  id: {ids}")
    print("  " + " | ".join(f"{i}:{p!r}" for i, p in zip(ids, pieces)))
    assert tok.decode(ids) == text, "decode(encode(x)) != x — так быть не должно!"
    print("  decode(encode(текст)) == текст ✓\n")


def explain(tok: BPETokenizer, text: str) -> None:
    """Повторяет логику _encode_chunk, но печатает каждый шаг."""
    for chunk in tok._compiled.findall(text):
        ids = list(chunk.encode("utf-8"))
        print(f"\nкусок {chunk!r}: {len(ids)} байт")
        print("  шаг 0:", " ".join(tok.id_to_token(i) for i in ids))
        step = 0
        while len(ids) >= 2:
            pairs = list(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: tok.merges.get(p, 1 << 62))
            if best not in tok.merges:
                break
            new_id = tok.merges[best]
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
            step += 1
            a, b = (tok.id_to_token(x) for x in best)
            print(
                f"  шаг {step}: склейка №{new_id - 256:<5} {a!r} + {b!r} -> {tok.id_to_token(new_id)!r:<14}"
                f" | " + " ".join(tok.id_to_token(i) for i in ids)
            )
        print(f"  итог: {len(ids)} токен(ов), id {ids}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/bpe_16k.json")
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    print(f"Загружен {args.tokenizer}: словарь {tok.vocab_size} токенов")

    if args.text is not None:
        (explain if args.explain else show)(tok, args.text)
        return

    print("Введите текст (?слово — пошагово, #id — токен по номеру, пустая строка — выход)")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        if line.startswith("?"):
            explain(tok, line[1:])
        elif line.startswith("#") and line[1:].isdigit():
            i = int(line[1:])
            print(f"  {i} -> {tok.id_to_token(i)!r}" if i in tok.vocab else "  нет такого id")
        else:
            show(tok, line)


if __name__ == "__main__":
    main()
