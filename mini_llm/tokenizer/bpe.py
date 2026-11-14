"""
Byte-level BPE токенизатор, написанный с нуля.

Идея (та же, что у GPT-2/GPT-4/Llama-3/Qwen):

1. Текст -> UTF-8 байты. Базовый словарь = 256 байт, поэтому любой текст
   кодируется без <unk>. Русская буква в UTF-8 занимает 2 байта,
   поэтому без обучения «привет» = 12 токенов.
2. Pre-tokenization: регуляркой режем текст на «куски» (слова, числа,
   пунктуация, пробелы). Слияния никогда не пересекают границу куска,
   поэтому не появится токен вида «т. В».
3. Обучение: считаем частоты соседних пар токенов во всём корпусе, самую
   частую пару склеиваем в новый токен, повторяем vocab_size - 256 раз.
4. Кодирование: в каждом куске применяем слияния в том порядке, в каком они
   были выучены (сначала с меньшим рангом).
"""

from __future__ import annotations # аннотации типов не вычисляются при запуске

import heapq # Куча: быстро находит самую частую пару при обучении.
import json
from collections import Counter # Словарь-счётчик. Считает, сколько раз встретился каждый кусок текста.
from collections import defaultdict # Словарь, который сам создаёт пустое значение для нового ключа (0 или пустое множество).
from pathlib import Path # удобная работа с путями к файлам
from typing import Iterable, Iterator

import regex as re

# Шаблон pre-tokenization из GPT-4 (cl100k_base). \p{L} — любая буква любого
# алфавита, поэтому кириллица обрабатывается так же, как латиница.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# служебные токены
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>", "<|system|>"]

Pair = tuple[int, int] # пара id

class BPETokenizer:
    def __init__(
        self,
        merges: dict[Pair, int] | None = None, # список выученных правил (~16000) склейки (обученный токенизатор)
            # ключ - пара id, значение - id нового токена
        special_tokens: dict[str, int] | None = None,
        pattern: str = GPT4_SPLIT_PATTERN,
    ):
        self.pattern = pattern
        self._compiled = re.compile(pattern)
        self.merges: dict[Pair, int] = merges or {}  # (a, b) -> id нового токена
        self.special_tokens: dict[str, int] = special_tokens or {}
        self._build_vocab()

    # vocab - таблица id -> байты
    def _build_vocab(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        for tok, idx in self.special_tokens.items():
            self.vocab[idx] = tok.encode("utf-8")
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}
        if self.special_tokens:
            alt = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            self._special_re = re.compile(f"({alt})")
        else:
            self._special_re = None
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # --------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
        pattern: str = GPT4_SPLIT_PATTERN,
        min_frequency: int = 2,
        verbose: bool = True,
    ) -> "BPETokenizer":
        """Обучить BPE на итерируемом наборе текстов (документов)."""
        special_tokens = DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens
        num_merges = vocab_size - 256 - len(special_tokens) # сколько склеек сделать
        if num_merges <= 0:
            raise ValueError("vocab_size должен быть больше 256 + число спецтокенов")

        compiled = re.compile(pattern)

        # 1) Считаем уникальные куски. Повторяющиеся слова обрабатываем один
        # раз с весом = частоте — это ускоряет обучение в сотни раз.
        chunk_counts: Counter[str] = Counter()
        n_docs = 0
        for text in texts:
            chunk_counts.update(compiled.findall(text))
            n_docs += 1
        if verbose:
            print(f"[bpe] документов: {n_docs:,}, уникальных кусков: {len(chunk_counts):,}")

        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk, c in chunk_counts.items():
            words.append(list(chunk.encode("utf-8")))
            freqs.append(c)
        del chunk_counts

        # 2) Начальные частоты пар + индекс «пара -> в каких словах встречается».
        pair_counts: dict[Pair, int] = defaultdict(int)
        where: dict[Pair, set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(wi)

        # 3) Max-куча с «ленивым» удалением: устаревшие записи пропускаем.
        #    При равной частоте выбирается меньшая пара — обучение детерминировано.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: dict[Pair, int] = {}
        next_id = 256
        log_every = max(1, num_merges // 20)

        while len(merges) < num_merges and heap:
            neg_c, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count != -neg_c:
                continue  # устаревшая запись
            if count < min_frequency:
                break
            new_id = next_id
            next_id += 1
            merges[pair] = new_id

            touched: set[Pair] = set()
            a, b = pair
            for wi in list(where.pop(pair, ())):
                w = words[wi]
                f = freqs[wi]
                # убираем старые пары слова
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                # склеиваем
                nw: list[int] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        nw.append(new_id)
                        i += 2
                    else:
                        nw.append(w[i])
                        i += 1
                words[wi] = nw
                # добавляем новые пары
                for p in zip(nw, nw[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    touched.add(p)

            pair_counts.pop(pair, None)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)

            if verbose and (len(merges) % log_every == 0 or len(merges) == num_merges):
                tok = _render(_bytes_of(new_id, merges))
                print(f"[bpe] merge {len(merges):>6}/{num_merges}: {tok!r} (частота {count:,})")

        specials = {tok: next_id + i for i, tok in enumerate(special_tokens)}
        return cls(merges=merges, special_tokens=specials, pattern=pattern)

    # --------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # пара с наименьшим рангом (выученная раньше всех)
            best = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, 1 << 62))
            new_id = self.merges.get(best)
            if new_id is None:
                break
            ids = _merge(ids, best, new_id)
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self._compiled.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Текст -> список id. Спецтокены в тексте распознаются, если allowed_special."""
        if not allowed_special or self._special_re is None:
            return self._encode_ordinary(text)
        out: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self._encode_ordinary(part))
        return out

    def encode_batch(self, texts: Iterable[str], allowed_special: bool = True) -> Iterator[list[int]]:
        for t in texts:
            yield self.encode(t, allowed_special=allowed_special)

    def decode(self, ids: Iterable[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def id_to_token(self, idx: int) -> str:
        """Человекочитаемое представление токена (для отладки)."""
        if idx in self.inverse_special:
            return self.inverse_special[idx]
        return _render(self.vocab[idx])

    # ---------------------------------------------------------- save / load
    def save(self, path: str | Path) -> None:
        """Сохраняет <path>.json (для загрузки) и <path>.vocab.txt (посмотреть глазами)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        merges_sorted = sorted(self.merges.items(), key=lambda kv: kv[1])
        data = {
            "type": "byte-level-bpe",
            "version": 1,
            "pattern": self.pattern,
            "merges": [[a, b] for (a, b), _ in merges_sorted],
            "special_tokens": self.special_tokens,
        }
        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with open(path.with_suffix(".vocab.txt"), "w", encoding="utf-8") as f:
            for idx in sorted(self.vocab):
                f.write(f"{idx}\t{self.id_to_token(idx)}\n")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        data = json.loads(path.read_text(encoding="utf-8"))
        merges = {(a, b): 256 + i for i, (a, b) in enumerate(data["merges"])}
        return cls(merges=merges, special_tokens=data["special_tokens"], pattern=data["pattern"])


# ---------------------------------------------------------------- helpers
def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    a, b = pair
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _bytes_of(idx: int, merges: dict[Pair, int]) -> bytes:
    inv = {v: k for k, v in merges.items()}

    def rec(i: int) -> bytes:
        if i < 256:
            return bytes([i])
        a, b = inv[i]
        return rec(a) + rec(b)

    return rec(idx)


def _render(b: bytes) -> str:
    """Байты токена -> строка; «обрезанные» UTF-8 последовательности показываем как <xNN>."""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(chr(x) if x < 128 else f"<x{x:02X}>" for x in b)
