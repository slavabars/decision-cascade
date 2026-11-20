# Паки весов на Hugging Face: скачать и запустить

**Русский** · [English](en/huggingface.md)

Обученные головы решений лежат в репозитории [`slavabars/decision-cascade-packs`](https://huggingface.co/slavabars/decision-cascade-packs). Это готовые веса для сервиса `decision-cascade`: скачал пак, указал энкодер — и сервис отвечает не случайными, а обученными головы.

## Что такое пак

Пак — это каталог с тремя файлами: `heads.pt` (веса голов и схемы вопросов), `metadata.json` (энкодер, размерность, подпись вопросов, метрики) и `temperature.json` (отчёт калибровки). Размер пака — десятки килобайт.

Важно: **это не `transformers`-модель**. Здесь нет `AutoModel.from_pretrained` и `pipeline("decision")`; энкодер (gpt2, distilgpt2, gpt2-medium) — отдельная модель на Хабе, её задаёт `DECISION_MODEL_ID`. В пак входят только головы, поэтому скачивание лёгкое, а веса энкодера вы получаете обычным путём через `transformers`.

## Какой пак выбрать

| Пак | Энкодер | `DECISION_FEATURE_DIM` | Датасет | Test accuracy | Вопросы |
| --- | --- | ---: | --- | ---: | --- |
| `support-gpt2` | `openai-community/gpt2` | 768 | синтетика (саппорт) | **0.926** | 6: срочность, команда, причина, накал, эскалация, удержание |
| `support-gpt2-medium` | `openai-community/gpt2-medium` | 1024 | синтетика (саппорт) | 0.914 | те же 6 |
| `support-distilgpt2` | `distilgpt2` | 768 | синтетика (саппорт) | 0.868 | те же 6 |
| `public-gpt2` | `openai-community/gpt2` | 768 | AG News | 0.880 | 3: topic, business, tech_level |
| `public-distilgpt2` | `distilgpt2` | 768 | AG News | 0.879 | те же 3 |
| `public-gpt2-medium` | `openai-community/gpt2-medium` | 1024 | AG News | 0.873 | те же 3 |
| `routerbench-gpt2` | `openai-community/gpt2` | 768 | RouterBench 0-shot | 0.548 | 3: escalate, route, benefit |

Числа — тестовый сплит пака, не гарантия на вашем домене: на своих данных калибруйте заново. Вопросы в запросе должны совпадать с набором пака (по смыслу и критериям), иначе сервис не найдёт подходящую подпись и останется без обученных голов. Паки `support-*` совпадают с набором вопросов сервиса по умолчанию, `public-*` — с `PUBLIC_QUESTIONS` (AG News), `routerbench-*` — с `ROUTERBENCH_QUESTIONS`.

## Скачать и запустить

Один пак:

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    "slavabars/decision-cascade-packs",
    revision="v1.5.0",
    allow_patterns=["packs/public-gpt2/*"],
)
print(path)
```

```bash
DECISION_MODEL_ID=openai-community/gpt2 \
DECISION_FEATURE_DIM=768 \
DECISION_CHECKPOINT_DIR=<path>/packs/public-gpt2 \
  uvicorn api.server.main:app --port 8000
```

Без ручного скачивания сервис может взять пак с Хаба сам (beta):

```bash
DECISION_PACK_REPO=slavabars/decision-cascade-packs@v1.5.0 \
DECISION_MODEL_ID=openai-community/gpt2 \
  uvicorn api.server.main:app --port 8000
```

- кэш — `DECISION_PACK_CACHE` (по умолчанию `~/.cache/decision-cascade/packs`); при повторном старте сеть не нужна;
- манифест и sha256 каждого файла проверяются; битый кэш перекачивается;
- ошибка загрузки — ошибка старта, а не тихий откат на случайные головы;
- `DECISION_PACK_REPO` и `DECISION_CHECKPOINT_DIR` взаимоисключающие.

Можно указать и весь каталог `packs/` — сервис выберет пак по вопросам, размерности и энкодеру. Но для конкретной модели надёжнее указать конкретный пак: у `public-gpt2` и `public-distilgpt2` одинаковые вопросы и размерность, различает их только `encoder` в метаданных.

Для `gpt2-medium` не забудьте `DECISION_FEATURE_DIM=1024` (у gpt2 и distilgpt2 — 768).

## Проверить, что пак загрузился

- `GET /info → engine.heads[].source` — путь к `heads.pt`, который реально используется;
- `GET /ready` — готовность сервиса; с `DECISION_UNTRAINED=refuse` сервис не подменит отсутствующий пак случайными головами;
- `trained: true` в ответе `POST /decisions`.

Если `trained: false`, чаще всего не совпали вопросы (подпись) или размерность энкодера.

## Ограничения

- `risk.json` и `conformal.json` в пак не входят: это отдельные артефакты, которые считаются на ваших сплитах (`eval.risk.run`, `eval.conformal.run`). Без них работают порог уверенности и benefit-политика.
- Метрики в карточке — конкретный датасет и тестовый сплит. На вашем домене они ничего не обещают.
- `heads.pt` — pickle; загрузчик читает его с `weights_only=True`.
- Пак не заменяет энкодер: без `DECISION_MODEL_ID` сервис использует hash-энкодер и не найдёт пак по размерности.

## Лицензии

Код и веса голов — Apache-2.0. Условия датасетов отдельные: AG News на Хабе помечен `license: unknown`, у RouterBench лицензия не декларирована, `support` — синтетический набор проекта. Файлы датасетов в репозиторий не входят, в карточке указаны только источники и метрики.

## Ещё

- Карточка со всеми паками и метриками: https://huggingface.co/slavabars/decision-cascade-packs
- Замеры и сравнение с конкурентами: [`benchmarks.md`](benchmarks.md)
