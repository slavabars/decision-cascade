# Паки весов на Hugging Face

**Русский** · [English](en/huggingface.md)

Обученные головы решений публикуются на Hugging Face Hub как отдельный репозиторий [`slavabars/decision-cascade-packs`](https://huggingface.co/slavabars/decision-cascade-packs). Репозиторий — это **не** `transformers`-модель: в нём нет `AutoModel.from_pretrained` и `pipeline("decision")`. Энкодер (gpt2, distilgpt2, gpt2-medium) остаётся отдельной моделью на Хабе и задаётся `DECISION_MODEL_ID`; пак содержит только головы, метаданные и отчёт калибровки.

## Что в репозитории

```text
README.md          # карточка: метрики, лицензии, инструкция
manifest.json      # версия, commit, размеры и sha256 каждого файла
packs/
  public-gpt2/{heads.pt, metadata.json, temperature.json}
  public-distilgpt2/
  public-gpt2-medium/
  support-gpt2/
  support-distilgpt2/
  support-gpt2-medium/
  routerbench-gpt2/
```

Датасеты (`dataset.jsonl`) не публикуются: они воспроизводятся из публичных наборов, а их хэш и размер сплитов есть в `metadata.json`. Паки `chess`, `maze` и `ru-rubert-tiny2` в репозиторий не входят: первые два — демо-примеры, у третьего неясны условия датасета.

## Скачать и запустить

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

`DECISION_CHECKPOINT_DIR` можно указать и на весь каталог `packs/` — сервис выберет пак по подписи вопросов и размерности. Но если в одном каталоге лежат паки с одинаковой подписью и одинаковой размерностью (`public-gpt2` и `public-distilgpt2`), победит первый по сортировке; для конкретной модели указывайте конкретный пак.

Скачанный каталог пака работает как `DECISION_CHECKPOINT_DIR` целиком: сервис распознаёт `metadata.json` + `heads.pt` прямо в корне. Это тот же формат, что в архиве GitHub Release, только без дублирующего `<signature>.pt`.

Проверить, что пак реально загрузился, можно через `GET /info → engine.heads[].source` или `GET /ready`; при `DECISION_UNTRAINED=refuse` сервис не подменит отсутствующий пак случайными головами.

## Один клик: `DECISION_PACK_REPO`

Вместо ручного `snapshot_download` сервис умеет сам взять паки с Хаба:

```bash
DECISION_PACK_REPO=slavabars/decision-cascade-packs@v1.5.0 \
DECISION_MODEL_ID=openai-community/gpt2 \
  uvicorn api.server.main:app --port 8000
```

- формат — `owner/name[@revision]`; без ревизии берётся `main`;
- кэш — `DECISION_PACK_CACHE` (по умолчанию `~/.cache/decision-cascade/packs`); если манифест и хэши в кэше сходятся, сеть не нужна;
- скачанный манифест проверяется по sha256 каждого файла; битый кэш перекачивается, а не подсовывается сервису;
- ошибка загрузки — ошибка старта, а не тихий откат на случайные головы;
- `DECISION_PACK_REPO` и `DECISION_CHECKPOINT_DIR` взаимоисключающие: непонятно, какой источник главнее, — лучше упасть на старте;
- `risk.json` и `conformal.json` в пак не входят, поэтому при включённом риск-бандле укажите `DECISION_RISK_MODEL_PATH` явно.

Режим помечен как beta: спецификация и поведение кэша могут уточняться, но выключен по умолчанию и ничего не меняет, пока переменная не задана.

## Публикация

Публикатор — `training/recipe/publish_hf.py`. Он собирает манифест и карточку из каталога паков, проверяет схему, хэши, размерность и лицензии датасетов, и только потом загружает.

```bash
# проверить паки и собрать манифест, ничего не загружая
python -m training.recipe.publish_hf --check

# staging-каталог: manifest.json, README.md, packs/<имя>/...
python -m training.recipe.publish_hf --dry-run --out dist/hf

# опубликовать и поставить тег-ревизию
HF_TOKEN=... python -m training.recipe.publish_hf \
  --publish --repo slavabars/decision-cascade-packs --revision v1.5.0

# сверить опубликованное с манифестом (нужен только доступ на чтение)
python -m training.recipe.publish_hf --verify \
  --repo slavabars/decision-cascade-packs --revision v1.5.0
```

Правила:

- аутентификация — `HF_TOKEN` в окружении или `huggingface-cli login` (токен ложится в `~/.cache/huggingface/token`); токен — секрет, не вставляйте его в чаты, issues и логи, а при утечке отзывайте;
- токен читается только из `HF_TOKEN` или локального логина и никогда не печатается;
- повторная публикация существующей ревизии не перезаписывает её: файлы сверяются по sha256, расхождение — ошибка «поднимите ревизию»;
- `--library-version` фиксирует, к какому релизу относятся паки (по умолчанию версия установленного пакета);
- `--exclude` убирает паки по имени каталога; по умолчанию исключены `chess`, `maze`, `ru-rubert-tiny2`;
- `--from-stage` публикует уже собранный staging (так делает CI, чтобы не пересобирать манифест).

## CI

`.github/workflows/publish-hf.yml` запускается вручную (`workflow_dispatch`). Это сознательно: паки не лежат в git, их артефакт — архив GitHub Release, поэтому workflow сначала скачивает `*-weights.tar.gz` указанного релиза, затем проверяет и собирает staging.

- `packs_release` — тег GitHub Release с архивом весов (по умолчанию `v1.5.0`);
- `revision` — тег в HF-репозитории; пусто — публикация только в `main`;
- `dry_run` — по умолчанию `true`: проверка и staging без записи на Хаб.

Запись включается только при `vars.HF_PUBLISH_ENABLED == 'true'` и заданном `secrets.HF_TOKEN`; без токена job завершается предупреждением. После загрузки smoke-шаг скачивает `public-gpt2` через `snapshot_download`, поднимает `DecisionService` с gpt2 и проверяет, что пак обучен и отвечает — то есть ловит и битый архив, и расхождение подписи вопросов.

## Лицензии и происхождение

Код и веса голов — Apache-2.0. Условия датасетов отдельные и указаны в карточке для каждого пака: AG News на Хабе помечен `license: unknown`, у RouterBench лицензия не декларирована, `support` — синтетический набор проекта. Файлы датасетов не перепубликуются; в манифесте остаются имена наборов, размеры сплитов и хэш исходного `dataset.jsonl`.

## Ограничения

- Пак не заменяет калибровку на вашем домене: метрики в карточке — тестовый сплит конкретного набора, а не гарантия на ваших данных.
- `risk.json` и `conformal.json` в пак не входят: это отдельные артефакты, которые считаются на ваших сплитах (`eval.risk.run`, `eval.conformal.run`).
- `heads.pt` — pickle; загрузчик читает его с `weights_only=True`, но переход на safetensors — отдельная задача, не условие публикации.
- Обновление пака — новая ревизия, а не перезапись тега: так `revision` остаётся воспроизводимой.
