import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.tokenizer import BPETokenizer  # noqa: E402

CORPUS = [
    "Мама мыла раму. Мама мыла раму снова и снова.",
    "Привет, мир! Привет всем, кто читает этот текст.",
    "Токенизатор учится склеивать частые пары байтов: мама, рама, мыла.",
    "Hello world, numbers 12345 and emoji 🙂 too.",
] * 20


def _train(vocab_size: int = 400) -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=vocab_size, verbose=False)


def test_roundtrip():
    tok = _train()
    for s in CORPUS[:4] + ["Совсем новый текст, которого не было: ёжик, ЪЬ, 中文, \t\n  пробелы"]:
        assert tok.decode(tok.encode(s)) == s


def test_compresses():
    tok = _train()
    s = CORPUS[0]
    assert len(tok.encode(s)) < len(s.encode("utf-8")) / 2


def test_special_tokens():
    tok = _train()
    eot = tok.special_tokens["<|endoftext|>"]
    ids = tok.encode("мама<|endoftext|>рама")
    assert eot in ids
    assert eot not in tok.encode("мама<|endoftext|>рама", allowed_special=False)
    assert tok.decode(ids) == "мама<|endoftext|>рама"


def test_vocab_size_and_ids():
    tok = _train(400)
    assert tok.vocab_size <= 400
    assert max(tok.vocab) == tok.vocab_size - 1  # id идут подряд без дыр


def test_save_load(tmp_path):
    tok = _train()
    tok.save(tmp_path / "tok")
    tok2 = BPETokenizer.load(tmp_path / "tok.json")
    s = "Мама мыла раму, привет мир!"
    assert tok.encode(s) == tok2.encode(s)
    assert (tmp_path / "tok.vocab.txt").exists()


def test_deterministic():
    assert _train().merges == _train().merges
