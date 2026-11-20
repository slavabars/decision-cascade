# Публикация паков на Hugging Face (для мейнтейнера)

**Русский** · [English](en/huggingface-publishing.md)

Страница для того, кто выпускает паки. Как их скачать и запустить — в [`huggingface.md`](huggingface.md).

Репозиторий: `slavabars/decision-cascade-packs` (публичный model repo). Ревизии — теги HF (`v1.5.0`); существующая ревизия не перезаписывается.

## Структура и манифест

```text
README.md          # карточка, генерируется publish_hf.py
manifest.json      # schema_version, library_version, git_commit, generated_at, packs[]
packs/<имя>/
  heads.pt         # веса голов + схемы вопросов + температуры
  metadata.json    # энкодер, dim, подпись вопросов, метрики train/val/test
  temperature.json # отчёт калибровки
```

`manifest.json` для каждого пака хранит `name`, `encoder`, `feature_dim`, `questions_signature`, `questions`, `dataset`, `dataset_source`, `dataset_license`, `splits`, `metrics.test` и `files` с `sha256` и `bytes`. Датасеты (`dataset.jsonl`) не публикуются: они воспроизводятся из публичных наборов, в метаданных остаётся хэш и размеры сплитов.

## Публикатор

`training/recipe/publish_hf.py` (входит в wheel):

```bash
# офлайн: схема, хэши, лицензии, staging
.venv/bin/python -m training.recipe.publish_hf --check --library-version v1.5.0
.venv/bin/python -m training.recipe.publish_hf --dry-run --out dist/hf

# публикация и проверка
.venv/bin/python -m training.recipe.publish_hf --publish \
  --repo slavabars/decision-cascade-packs --revision v1.5.0
.venv/bin/python -m training.recipe.publish_hf --verify \
  --repo slavabars/decision-cascade-packs --revision v1.5.0
```

- `--packs` — каталог паков (по умолчанию `checkpoints/packs`);
- `--exclude` — паки по имени каталога; по умолчанию `chess`, `maze`, `ru-rubert-tiny2`;
- `--library-version` — к какому релизу относятся паки (в CI — тег архива весов);
- `--commit` — git commit сборки паков, если известен;
- `--from-stage` — публикация уже собранного staging (так делает CI);
- `--check`, `--dry-run`, `--from-stage`, `--verify` не пишут на Хаб без `--publish`.

Правила:

- аутентификация — `HF_TOKEN` или локальный `hf auth login` (CLI `huggingface_hub` 2.x; модуля `huggingface_hub.commands` больше нет). Токен не печатается и не коммитится; при утечке — отозвать;
- существующий тег не перезаписывается: `publish` сверяет sha256 и при расхождении требует новую ревизию;
- после загрузки `publish` сам вызывает `--verify` (с повторами на распространение тега).

## CI

`.github/workflows/publish-hf.yml` — только `workflow_dispatch`: паки не в git, архив весов берётся из GitHub Release.

- входы: `packs_release` (по умолчанию `v1.5.0`), `revision`, `dry_run` (по умолчанию `true`);
- job `check` всегда скачивает `*-weights.tar.gz`, собирает staging и кладёт артефакт `hf-staging`;
- job `publish` включается при `vars.HF_PUBLISH_ENABLED == 'true'` и заданном `secrets.HF_TOKEN`; публикует из артефакта и запускает smoke;
- smoke скачивает `public-gpt2` через `snapshot_download`, поднимает `DecisionService` с `DECISION_MODEL_ID=openai-community/gpt2` и проверяет `is_trained` и один ответ.

Настройка репозитория:

```bash
gh variable set HF_PUBLISH_ENABLED --body true -R slavabars/decision-cascade
gh secret set HF_TOKEN -R slavabars/decision-cascade < ~/.cache/huggingface/token
```

## Чек-лист нового пака или ревизии

1. Пак обучен и откалиброван, `metadata.json` содержит энкодер, dim, подпись и метрики.
2. `publish_hf.py --check` проходит; предупреждения о лицензиях датасетов просмотрены.
3. `--dry-run` собирает staging, пак из staging поднимается как `DECISION_CHECKPOINT_DIR`.
4. Для новой ревизии — новый тег; старый не трогать.
5. После публикации `--verify` и smoke зелёные.
6. Карточка и `docs/huggingface.md` не разошлись по числам (метрики в карточке генерируются из манифеста).

## Частые ошибки

| Симптом | Причина и что делать |
| --- | --- |
| `already published and matches` | Это не ошибка: тег и файлы совпадают, ничего не перезаписано |
| `published packs differ from the local build` | Изменились файлы при том же теге: поднимите `--revision` |
| `not authenticated` | Нет `HF_TOKEN` и локального логина: `hf auth login` |
| `unsafe pack name in manifest` | В манифесте путь с `/`, `\` или `..` — публикация отклонена |
| `sha256 mismatch` | Битый кэш или архив: перекачать; `publish --verify` покажет файл |
| `trained: false` у пользователя | Не совпали вопросы/размерность/энкодер; сервис сверяет подпись, dim и `encoder` |

Тесты: `tests/test_publish_hf.py`, `tests/test_packs_hub.py`, `tests/test_multidomain.py`. Контракт пользователя — [`huggingface.md`](huggingface.md).
