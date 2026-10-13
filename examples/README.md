# Примеры

Готовые скрипты «на все случаи». Большинство работают **в процессе, без сервера и
без скачивания моделей** (детерминированный hash-энкодер), поэтому запускаются
мгновенно. Все команды выполняются из корня репозитория.

> По умолчанию головы необучены: числа демонстрируют конвейер, а не качество
> решений. Для реального качества обучите головы (`training/`) или возьмите пак
> весов — см. `ops/weight_pack.py`.

Общие хелперы примеров — `_common.py` (сборка сервиса, наборы вопросов, печать,
JSONL/CSV). Примеры подключают его как `from examples._common import ...`.

Условные обозначения: **Сервер** — нужен ли запущенный API (`uvicorn …`), или
пример поднимает его сам, или обходится без него (in-process).

---

## Быстрый старт

### `quickstart/hello.py` — сервер: нет

Минимальный вход: одно текстовое состояние и шесть вопросов всех трёх типов
(`noul` / `choice` / `score`), посчитанных in-process. Ничего скачивать не нужно.

```bash
.venv/bin/python examples/quickstart/hello.py
```

Печатает исходное состояние и человекочитаемый блок решений: для `noul` —
вероятность и уверенность, для `choice` — выбор и распределение по опциям, для
`score` — балл и распределение по уровням.

### `quickstart/quickstart.sh` — сервер: поднимает сам

Полный HTTP-цикл: запускает `uvicorn`, ждёт готовности, делает `POST /decisions`
по трём вопросам, печатает JSON и останавливает сервер. Удобно как первая
проверка, что окружение собрано.

```bash
bash examples/quickstart/quickstart.sh
```

Переменные: `PYTHON` (по умолчанию `.venv/bin/python`), `PORT` (`8000`),
`MODEL_ID` (если задать — поднимет сервер с реальным энкодером).

### `quickstart/http_client.py` — сервер: да

Работа с сервисом через Python SDK (`DecisionClient`): те же шесть вопросов
разного типа, но по сети. Показывает, как получить решения и напечатать
вероятности по каждой опции.

```bash
.venv/bin/python -m uvicorn api.server.main:app --port 8000
.venv/bin/python examples/quickstart/http_client.py
```

Печатает разбор обращения (команда, причина, накал, эскалация, риск оттока) и
`schema_version` ответа.

---

## Батчи и данные

### `batch/inprocess_batch.py` — сервер: нет

Батч `DecisionService.decide_many`: много обращений за один прогон энкодера,
без HTTP. Печатает компактную таблицу `id / команда / срочность / накал /
эскалация` и затраченное время.

```bash
.venv/bin/python examples/batch/inprocess_batch.py --n 8
```

Флаги: `--n` (число обращений), `--seed`.

### `batch/csv_triage.py` — сервер: нет

Готовая ETL: читает CSV с колонкой текста, прогоняет все строки одним батчем и
дописывает колонки с решениями (`команда/срочность/эскалация/накал`). Без
`--input` генерирует демонстрационную выгрузку.

```bash
.venv/bin/python examples/batch/csv_triage.py --n 20
.venv/bin/python examples/batch/csv_triage.py --input tickets.csv --output out.csv
```

Флаги: `--input`, `--output` (по умолчанию `data/examples/triage.csv`),
`--text-column` (`text`), `--n`, `--seed`.

### `batch/jsonl_batch.py` — сервер: да

Серверный батч через `POST /decisions/batch`: строки JSONL `{"id","state"}`
отправляются чанками (до 256 за запрос), результат пишется в JSONL с решениями.

```bash
.venv/bin/python -m uvicorn api.server.main:app --port 8000
.venv/bin/python examples/batch/jsonl_batch.py --n 10
```

Флаги: `--base-url`, `--input`, `--output` (`data/examples/batch.jsonl`),
`--n`, `--seed`.

---

## Протокол SystemOne

### `protocol/systemone.sh` — сервер: поднимает сам

`POST /v1/systemone` — тот же протокол, что у Clef/Clef-Flash, vLLM, SGLang.
Показывает вопросы, заданные только через `criteria` (опции/уровни выводятся
автоматически), и типизированную ошибку `MODALITY_NOT_SUPPORTED` при передаче
`images`, которую локальный текстовый бэкенд не поддерживает.

```bash
bash examples/protocol/systemone.sh
```

### `protocol/systemone_client.py` — сервер: да

Минимальный клиент протокола на голом `httpx` (без нашего SDK) — класс
`SystemOneClient`. Тем же кодом вызывается и наш сервер, и внешняя модель:
достаточно поменять `--base-url` и `--model`.

```bash
.venv/bin/python -m uvicorn api.server.main:app --port 8000
.venv/bin/python examples/protocol/systemone_client.py
.venv/bin/python examples/protocol/systemone_client.py --base-url http://127.0.0.1:30000 --model clef-flash
```

Флаги: `--base-url`, `--model`.

### `protocol/gateway.sh` — сервер: поднимает сам

Шлюз `DECISION_BACKEND=http`: поднимает «внешний» сервис (порт 8001) и наш
сервер как шлюз (8000), который проксирует к нему. Показывает, как подключить
любую внешнюю SystemOne-модель, не меняя протокол.

```bash
bash examples/protocol/gateway.sh
```

Переменные: `UPSTREAM_PORT` (`8001`), `GATEWAY_PORT` (`8000`).

---

## Каскады

### `cascade/hash_to_llm.py` — сервер: нет

Двухуровневый каскад in-process на `CascadeBackend`: дешёвый локальный бэкенд
решает всё уверенно, остальное эскалируется в fallback. Роль «LLM» играет
заглушка `StubFallback` — она считает вызовы. Показывает `usage.escalated`.

```bash
.venv/bin/python examples/cascade/hash_to_llm.py --threshold 0.6
```

Флаги: `--threshold` (порог уверенности), `--seed`. В продакшене fallback —
это `HttpSystemOneBackend` (см. `protocol/gateway.sh`).

### `cascade/client_fallback.py` — сервер: да (два)

Клиентский каскад: запрашивает primary, считает минимальную уверенность по
ответам и, если она ниже порога, повторяет запрос к fallback. Полезно, когда
primary и fallback разнородны (наш сервер + внешний LLM).

```bash
.venv/bin/python -m uvicorn api.server.main:app --port 8000
.venv/bin/python -m uvicorn api.server.main:app --port 8001
.venv/bin/python examples/cascade/client_fallback.py --fallback http://localhost:8001
```

Флаги: `--primary`, `--fallback`, `--threshold`.

### `cascade/server_cascade.sh` — сервер: поднимает сам

Серверный каскад: апстрим (8001) + каскадный сервер (8000) с порогом. Низко­
уверенные вопросы каскад эскалирует сам, ответ содержит `usage.escalated`.

```bash
bash examples/cascade/server_cascade.sh
```

Переменные: `UPSTREAM_PORT` (`8001`), `CASCADE_PORT` (`8000`),
`THRESHOLD` (`0.9`).

---

## Vision / мультимодальность

### `vision/make_sample_images.py` — сервер: нет

Генерирует синтетические изображения (фигуры разных цветов/типов) и манифест
`manifest.jsonl` с метками. Материал для остальных vision-примеров.

```bash
.venv/bin/python examples/vision/make_sample_images.py --n 8
```

Флаги: `--n`, `--out` (`data/examples/vision/images`), `--seed`.

### `vision/image_single.py` — сервер: нет

Одно изображение → решения: текст состояния и картинка идут в один запрос,
признаки сливаются с весом `media_weight`. Если файла нет — сгенерирует его.

```bash
.venv/bin/python examples/vision/image_single.py
.venv/bin/python examples/vision/image_single.py --image shot.png --weight 0.8
```

Флаги: `--image`, `--weight`. По умолчанию `HashVisionEncoder` (офлайн); CLIP
включается переменной `DECISION_VISION_MODEL_ID` на стороне сервера.

### `vision/image_folder.py` — сервер: нет

Пакетная обработка папки PNG: решения по каждому файлу плюс агрегат
(распределение по типу и число «критичных»). При пустой папке изображения
генерируются автоматически.

```bash
.venv/bin/python examples/vision/image_folder.py
```

Флаги: `--dir`, `--weight`, `--seed`, `--show-first` (сколько файлов распечатать
подробно).

### `vision/video_frames.py` — сервер: нет

Видео → кадры: из картинок собирается псевдо-клип, запрос передаёт `videos` и
`media_kwargs={"max_frames": N}`. Показывает, как обрабатывать видеопоток
(`HashVisionEncoder`; CLIP поддерживает только изображения).

```bash
.venv/bin/python examples/vision/video_frames.py
```

Флаги: `--dir`, `--clip`, `--frames` (число кадров), `--weight`.

### `vision/vision_offline.py` — сервер: нет

Минимум по мультимодальности: список модальностей сервиса и влияние
`media_weight` (`0.0` / `0.5` / `1.0`) на ответ. Детерминированно, без сети —
годится для тестов и CI.

```bash
.venv/bin/python examples/vision/vision_offline.py
```

---

## Интеграции

### `integrations/chat_agent.py` — сервер: да

Интеграция решений в чат-агента через SDK + `route_ticket`: по типизированным
решениям выбираются команда, приоритет, эскалация и тон ответа. При
`--fallback` включён клиентский каскад — низкоуверенное обращение переспрашивает
внешний SystemOne-сервис.

```bash
.venv/bin/python -m uvicorn api.server.main:app --port 8000
.venv/bin/python examples/integrations/chat_agent.py
.venv/bin/python examples/integrations/chat_agent.py --fallback http://remote:8080
```

Флаги: `--base-url`, `--fallback`.

### `integrations/telegram_bot.py` — сервер: нет

Telegram-бот: long polling `getUpdates`, ответ формируется решениями
in-process, отправляется `sendMessage`. Без токена запускается dry-run на
демо-сообщении (наружу ничего не уходит).

```bash
.venv/bin/python examples/integrations/telegram_bot.py
TELEGRAM_BOT_TOKEN=... .venv/bin/python examples/integrations/telegram_bot.py --max-updates 5
```

Флаги: `--token` (или env `TELEGRAM_BOT_TOKEN`), `--max-updates` (`0` —
бесконечно).

### `integrations/discord_bot.py` — сервер: нет

Ответ в канал Discord через REST (без websocket-зависимостей). Без токена и
канала — dry-run; с `--send` реально отправляет сообщение.

```bash
.venv/bin/python examples/integrations/discord_bot.py
DISCORD_BOT_TOKEN=... DISCORD_CHANNEL_ID=... .venv/bin/python examples/integrations/discord_bot.py --send
```

Флаги: `--token`, `--channel` (или env `DISCORD_BOT_TOKEN`,
`DISCORD_CHANNEL_ID`), `--send`.

### `integrations/slack_app.py` — сервер: нет / `--serve`

Слэш-команда `/triage` на FastAPI: тело (form-urlencoded) разбирается
стандартным `urllib.parse`, ответ формируется решениями. Без `--serve` — демо
через `TestClient`.

```bash
.venv/bin/python examples/integrations/slack_app.py           # демо
.venv/bin/python examples/integrations/slack_app.py --serve   # uvicorn :8000
```

Флаги: `--serve`, `--port`. URL команды в Slack: `https://<host>/slack/triage`.

### `integrations/webhook_fastapi.py` — сервер: нет

Готовый `APIRouter` (`/decisions/analyze`), который подключается к любому
своему FastAPI-приложению через `app.include_router(decision_router)`. Демо —
через `TestClient`.

```bash
.venv/bin/python examples/integrations/webhook_fastapi.py
```

### `integrations/openai_tools.py` — сервер: нет

Решения как инструмент (function calling): JSON-схема `analyze_ticket`,
симуляция tool-call от «модели», выполнение и результат в формате
tool-сообщения. Годится как точка входа для OpenAI/Anthropic/LangChain.

```bash
.venv/bin/python examples/integrations/openai_tools.py
```

### `integrations/email_triage.py` — сервер: нет

Письмо → карточка триажа: команда, приоритет, эскалация, тон, накал, риск
оттока, черновик ответа и обоснование.

```bash
.venv/bin/python examples/integrations/email_triage.py
.venv/bin/python examples/integrations/email_triage.py --subject "Двойное списание" --body "Верните деньги"
```

Флаги: `--subject`, `--body`.

### `integrations/support_copilot.py` — сервер: нет

Карточка рабочего места агента поддержки: маршрут, предлагаемый макрос ответа,
следующие шаги (эскалация, business-critical, риск оттока) и черновик.

```bash
.venv/bin/python examples/integrations/support_copilot.py
```

### `integrations/moderation.py` — сервер: нет

Модерация контента на своём наборе вопросов (токсичность / категория /
серьёзность) → действие `allow` / `review` / `block`. Показывает, как заменять
вопросы под задачу. В конце — пометка про необученные головы.

```bash
.venv/bin/python examples/integrations/moderation.py
```

### `integrations/sentiment_router.py` — сервер: нет

Маршрутизация по тональности и срочности → очередь (`VIP-эскалация`, старшая
линия, обычная линия, запрос отзыва). Ещё один способ связать решения с
действием.

```bash
.venv/bin/python examples/integrations/sentiment_router.py
```

---

## Эксплуатация и веса

### `ops/local_heads.py` — сервер: нет

In-process работа без HTTP плюс сохранение и загрузка голов: `save_heads`
пишет чекпоинт, новый сервис подхватывает его из каталога (сид уже не влияет).

```bash
.venv/bin/python examples/ops/local_heads.py
```

Флаги: `--dir` (`data/examples/heads`).

### `ops/weight_pack.py` — сервер: нет

Запуск с обученным паком весов: реальный трансформер-энкодер + обученные
головы, заданные явно (`encoder + checkpoint_dir`). Если пака нет — печатает
подсказку, как его получить, и завершается успешно.

```bash
.venv/bin/python examples/ops/weight_pack.py
.venv/bin/python examples/ops/weight_pack.py --pack checkpoints/packs/public-gpt2 --model openai-community/gpt2 --dim 768
```

Флаги: `--pack`, `--model`, `--dim`, `--device`, `--dtype`, `--kind`. Паки —
из релиза `v0.2.0` (см. README).

### `ops/device_select.py` — сервер: нет

Выбор устройства и типа вычислений: `resolve_device` / `resolve_dtype` /
`check_dtype_device` (та же логика, что в сервере) и замер кодирования
небольшого батча.

```bash
.venv/bin/python examples/ops/device_select.py
.venv/bin/python examples/ops/device_select.py --device cuda --dtype bfloat16
```

Флаги: `--device` (`auto` / `cpu` / `cuda` / `cuda:0` / `mps`), `--dtype`
(`float32` / `float16` / `bfloat16`), `--n`.

### `ops/health_check.sh` — сервер: поднимает сам

Проверка живости: поднимает сервер, ждёт готовности, проверяет `/openapi.json`
и делает smoke-запрос к `/decisions`. Возвращает ненулевой код, если сервер не
ответил — удобно для CI и мониторинга.

```bash
bash examples/ops/health_check.sh
```

Переменные: `PYTHON` (`.venv/bin/python`), `PORT` (`8000`).

---

## Smoke-проверка

Быстрая проверка, что примеры действительно запускаются (offline-примеры
выполняются в процессе, сетевые — через `--help`):

```bash
.venv/bin/python -m pytest tests/test_examples.py -q
```
