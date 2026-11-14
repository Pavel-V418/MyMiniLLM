# MyMiniLLM

Своя небольшая LLM с нуля → финансовый ассистент. Код хранится здесь, тяжёлое обучение идёт на Kaggle.

## Структура

```
mini_llm/
  tokenizer/bpe.py        # byte-level BPE с нуля: train / encode / decode / save / load
  data.py                 # чтение корпуса (документы разделены <|endoftext|>)
scripts/
  prepare_corpus.py       # кусок русского корпуса с Hugging Face -> data/corpus/{train,val}.txt
  train_tokenizer.py      # обучение BPE -> artifacts/tokenizer/bpe_16k.json
  compare_tokenizers.py   # сравнение с GPT-2/4/4o и Qwen2.5 -> artifacts/tokenizer/comparison.md
tests/test_bpe.py         # roundtrip, спецтокены, save/load, детерминизм
```

## Этап 1: свой токенизатор

```bash
pip install -r requirements.txt
python scripts/prepare_corpus.py --mb 20          # русская Википедия, ~20 МБ
python scripts/train_tokenizer.py --vocab-size 16000 --max-mb 20
python scripts/compare_tokenizers.py
pytest -q
```

Обучение на 20 МБ занимает около 1 минуты на ноутбуке.

### Результат (16k словарь, 20 МБ Википедии, валидация 230k символов)

| Токенизатор | Словарь | Символов/токен ↑ | Токенов/слово ↓ |
|---|---:|---:|---:|
| символы (этап 0) | ~150 | 1.00 | 7.66 |
| байты UTF-8 | 256 | 0.56 | 13.65 |
| **наш BPE** | **16 000** | **3.52** | **2.18** |
| GPT-2 | 50 257 | 0.96 | 7.97 |
| GPT-4 (cl100k) | 100 277 | 2.00 | 3.84 |
| GPT-4o (o200k) | 200 019 | 3.20 | 2.39 |
| Qwen2.5 | 151 665 | 2.56 | 2.99 |

Выводы:
- По сравнению с посимвольным токенизатором текст сжимается в **3.5 раза**: при том же контексте модель видит в 3.5 раза больше текста.
- Словарь в 10 раз меньше, чем у GPT-4o, а на русском мы плотнее: весь наш словарь потрачен на русский, у них на 100+ языков и код.
- GPT-2 обучен почти только на английском: каждая русская буква у него = 1–2 токена, русский текст выходит длиннее, чем в посимвольной кодировке.
- Слабое место нашего токенизатора: редкие символы (`₽` разбит на 3 байта) и числа. Если на этапе 5 в данных будет много сумм, стоит учить токенизатор на корпусе с транзакциями.
- Подсчёт букв: `пересмотреть` → `пер · ес · мот · реть`. Модель видит 4 id, а не 12 букв, поэтому ей трудно посчитать «р».

## Этап 0: mini-GPT

GPT-2-подобная модель с нашим BPE-токенизатором.

| Файл | Что делает |
|---|---|
| `mini_llm/model.py` | модель: эмбеддинги, causal self-attention, MLP, блоки, генерация |
| `mini_llm/configs.py` | пресеты `debug` (ноутбук, CPU) и `small` (Kaggle, ~17M параметров) |
| `scripts/tokenize_corpus.py` | `data/corpus/*.txt` → `data/tokens/*.bin` (uint16, на всех ядрах) |
| `scripts/train.py` | цикл обучения: AdamW, warmup + cosine LR, mixed precision, чекпоинты, W&B |
| `scripts/generate.py` | генерация текста из чекпоинта |
| `kaggle/stage0_train.ipynb` | ноутбук для Kaggle: clone → данные → обучение → генерация → график |
| `tests/test_model.py` | формы, каузальность, weight tying, переобучение на одном батче |

### Порядок работы

На ноутбуке, чтобы проверить код (~1–2 минуты):

```bash
python scripts/tokenize_corpus.py
python scripts/train.py --config debug
python scripts/generate.py --ckpt checkpoints/debug/best.pt --prompt "Москва — это"
```

На Kaggle — полное обучение: загрузите `kaggle/stage0_train.ipynb` (File → Import Notebook), проверьте `REPO_URL`, включите GPU и Internet, Run All.

Параметры пресета переопределяются из командной строки: `--max-steps 2000 --lr 6e-4 --n-layer 8`.
Обрыв сессии: тот же запуск с `--resume` продолжит с `last.pt`.
