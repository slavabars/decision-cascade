# Примеры

Готовые скрипты «на все случаи». Большинство работают **в процессе, без сервера и
без скачивания моделей** (детерминированный hash-энкодер), поэтому запускаются
мгновенно. Все команды выполняются из корня репозитория.

> По умолчанию головы необучены: числа демонстрируют конвейер, а не качество
> решений. Для реального качества обучите головы (`training/`) или возьмите пак
> весов — см. `ops/weight_pack.py`.

Общие хелперы примеров — `_common.py` (сервис, наборы вопросов, печать, CSV/JSONL).

## Быстрый старт

| Пример | Что показывает | Сервер | Запуск |
| --- | --- | --- | --- |
| `quickstart/hello.py` | Минимум: один вход, вопросы noul/choice/score, in-process | нет | `.venv/bin/python examples/quickstart/hello.py` |
| `quickstart/quickstart.sh` | Поднять сервер и сделать запрос к `/decisions` | поднимает сам | `bash examples/quickstart/quickstart.sh` |
| `quickstart/http_client.py` | Python SDK (`DecisionClient`) к серверу | да | `.venv/bin/python examples/quickstart/http_client.py` |

## Батчи и данные

| Пример | Что показывает | Сервер | Запуск |
| --- | --- | --- | --- |
| `batch/inprocess_batch.py` | Батч in-process (`decide_many`), без HTTP | нет | `.venv/bin/python examples/batch/inprocess_batch.py --n 8` |
| `batch/csv_triage.py` | CSV → CSV: готовый триаж выгрузки | нет | `.venv/bin/python examples/batch/csv_triage.py --n 20` |
| `batch/jsonl_batch.py` | Серверный батч `/decisions/batch` | да | `.venv/bin/python examples/batch/jsonl_batch.py --n 10` |

## Протокол SystemOne

| Пример | Что показывает | Сервер | Запуск |
| --- | --- | --- | --- |
| `protocol/systemone.sh` | `/v1/systemone`: criteria-only вопросы и ошибка модальности | поднимает сам | `bash examples/protocol/systemone.sh` |
| `protocol/systemone_client.py` | Минимальный httpx-клиент протокола | да | `.venv/bin/python examples/protocol/systemone_client.py` |
| `protocol/gateway.sh` | Шлюз `DECISION_BACKEND=http` к внешней модели | поднимает сам | `bash examples/protocol/gateway.sh` |

## Каскады

| Пример | Что показывает | Сервер | Запуск |
| --- | --- | --- | --- |
| `cascade/hash_to_llm.py` | Двухуровневый каскад in-process (`CascadeBackend`) | нет | `.venv/bin/python examples/cascade/hash_to_llm.py` |
| `cascade/client_fallback.py` | Клиентская эскалация по уверенности | да (два) | `.venv/bin/python examples/cascade/client_fallback.py --fallback http://localhost:8001` |
| `cascade/server_cascade.sh` | Серверный каскад local-first + fallback | поднимает сам | `bash examples/cascade/server_cascade.sh` |

## Vision / мультимодальность

| Пример | Что показывает | Сервер | Запуск |
| --- | --- | --- | --- |
| `vision/make_sample_images.py` | Сгенерировать тестовые картинки | нет | `.venv/bin/python examples/vision/make_sample_images.py --n 8` |
| `vision/image_single.py` | Одно изображение → решения | нет | `.venv/bin/python examples/vision/image_single.py` |
| `vision/image_folder.py` | Папка изображений → решения + агрегат | нет | `.venv/bin/python examples/vision/image_folder.py` |
| `vision/video_frames.py` | Видео → кадры (`max_frames`) | нет | `.venv/bin/python examples/vision/video_frames.py` |
| `vision/vision_offline.py` | `HashVisionEncoder`, влияние `media_weight` | нет | `.venv/bin/python examples/vision/vision_offline.py` |

> CLIP подключается переменной `DECISION_VISION_MODEL_ID` (только изображения).
> Для видео используется `HashVisionEncoder`.

## Интеграции

| Пример | Что показывает | Сервер | Запуск |
| --- | --- | --- | --- |
| `integrations/chat_agent.py` | Маршрутизация тикета + клиентский каскад | да | `.venv/bin/python examples/integrations/chat_agent.py` |
| `integrations/telegram_bot.py` | Telegram-бот (dry-run без токена) | нет | `TELEGRAM_BOT_TOKEN=... .venv/bin/python examples/integrations/telegram_bot.py` |
| `integrations/discord_bot.py` | Discord-уведомления (dry-run без токена) | нет | `DISCORD_BOT_TOKEN=... DISCORD_CHANNEL_ID=... .venv/bin/python examples/integrations/discord_bot.py --send` |
| `integrations/slack_app.py` | Слэш-команда `/triage` (FastAPI) | нет / `--serve` | `.venv/bin/python examples/integrations/slack_app.py` |
| `integrations/webhook_fastapi.py` | Роутер решений в своём FastAPI-приложении | нет | `.venv/bin/python examples/integrations/webhook_fastapi.py` |
| `integrations/openai_tools.py` | Решения как function-calling инструмент | нет | `.venv/bin/python examples/integrations/openai_tools.py` |
| `integrations/email_triage.py` | Письмо → триаж + черновик ответа | нет | `.venv/bin/python examples/integrations/email_triage.py` |
| `integrations/support_copilot.py` | Карточка агента: маршрут, макрос, шаги | нет | `.venv/bin/python examples/integrations/support_copilot.py` |
| `integrations/moderation.py` | Модерация: allow/review/block | нет | `.venv/bin/python examples/integrations/moderation.py` |
| `integrations/sentiment_router.py` | Тональность → очередь | нет | `.venv/bin/python examples/integrations/sentiment_router.py` |

## Эксплуатация и веса

| Пример | Что показывает | Сервер | Запуск |
| --- | --- | --- | --- |
| `ops/local_heads.py` | In-process + сохранение/загрузка голов | нет | `.venv/bin/python examples/ops/local_heads.py` |
| `ops/weight_pack.py` | Запуск с обученным паком (реальный энкодер) | нет | `.venv/bin/python examples/ops/weight_pack.py` |
| `ops/device_select.py` | Выбор `cpu`/`cuda`/dtype и замер | нет | `.venv/bin/python examples/ops/device_select.py` |
| `ops/health_check.sh` | Живость сервера + smoke `/decisions` | поднимает сам | `bash examples/ops/health_check.sh` |

## Smoke-проверка

Быстрая проверка, что offline-примеры действительно запускаются:

```bash
.venv/bin/python -m pytest tests/test_examples.py -q
```
