# Troubleshooting

Типовые ошибки сервиса: что означает код, откуда он берётся и что делать.
Для эксплуатации (systemd, upgrade/rollback) — [`deploy.md`](deploy.md).

## 1. Быстрая триада

```bash
curl -fsS http://127.0.0.1:8080/health    # процесс жив?
curl -sS  http://127.0.0.1:8080/ready     # готов обслуживать? (503 + reasons)
curl -fsS http://127.0.0.1:8080/info      # бэкенд, модальности, движок
journalctl -u decision-api@api -n 100 --no-pager
```

`/health` отвечает всегда, пока жив процесс. Если он молчит — смотрите
`systemctl status decision-api@api.service` и порт (`ss -ltnp | grep 8080`).
`/ready` — единственный правильный источник для балансировщика: `503` означает
«не шли трафик».

## 2. Коды ошибок API

Все ошибки — JSON `{"error": {"code": ..., "message": ...}}`.

| Код | HTTP | Когда |
| --- | --- | --- |
| `UNTRAINED_HEADS` | 409 | под набор вопросов нет обученного пака, а `DECISION_UNTRAINED=refuse`/`auto` с явным `DECISION_CHECKPOINT_DIR` |
| `MODALITY_NOT_SUPPORTED` | 422 | в `/v1/systemone` пришли `images`/`videos`, а бэкенд их не умеет |
| `BACKEND_UNAVAILABLE` | 502 | внешний SystemOne не ответил за бюджет `DECISION_BACKEND_TIMEOUT` (все retry исчерпаны) |
| `FEEDBACK_UNSUPPORTED` | 409 | `/v1/feedback` вызван без каскада, без `DECISION_CASCADE_RISK_BUDGET` или с `DECISION_RISK_FEW_SHOT_N < 2` |
| `INVALID_QUESTION` | 422 | вопрос из `/v1/systemone` не удалось преобразовать в спеку |
| `422` (FastAPI) | 422 | тело запроса не прошло pydantic-валидацию (см. §4) |

## 3. `UNTRAINED_HEADS`

```json
{"error": {"code": "UNTRAINED_HEADS", "signature": "d4dcbc937478268a",
           "checkpoint_dir": "checkpoints"}}
```

Головы есть, но обученного пака под **подпись вопросов** нет. Подпись считается
по именам, типам, инструкциям, вариантам и критериям — изменение любого из них
даёт другую подпись.

Что проверить:

```bash
ls checkpoints/packs/*/metadata.json
.venv/bin/python - <<'EOF'
import json, glob
for path in glob.glob("checkpoints/packs/*/metadata.json"):
    m = json.load(open(path))
    print(path, m.get("specs"), m.get("questions_signature"))
EOF
```

- `DECISION_CHECKPOINT_DIR` указывает на каталог с паками?
- В запросе те же `instructions`/`options`/`criteria`, что при обучении?
- Пак лежит в `checkpoints/packs/<name>/`, а не в `checkpoints/`?

Варианты:

- переобучить головы под текущие вопросы (`python -m training.recipe.train_heads ...`);
- положить пак с совпадающей подписью;
- для демо/отладки разрешить необученные головы: `DECISION_UNTRAINED=allow`
  (в ответе будет `trained: false`, числа случайные).

Лог при старте подсказывает точнее:

```
WARNING no checkpoint <sig>.pt for questions signature <sig>; serving untrained heads
```

## 4. `422` от pydantic

Частые причины:

- `options`/`levels` с дублями — варианты обязаны быть уникальны;
- `criteria` у `score` не совпадает по длине с `levels`;
- `states` в `/decisions/batch` содержит пустую строку;
- `confidence`/`probabilities` в `/v1/feedback` вне `[0, 1]` или не суммируются
  в 1 (примеры с негодным решением не принимаются, чтобы не учить на них).

Тело ответа FastAPI содержит `detail[].loc` — путь до поля.

## 5. `MODALITY_NOT_SUPPORTED`

```json
{"error": {"code": "MODALITY_NOT_SUPPORTED", "supported": ["text"],
           "requested": ["image"]}}
```

Локальный бэкенд без vision-энкодера принимает только текст. Включить CLIP:

```bash
DECISION_VISION_MODEL_ID=openai/clip-vit-base-patch32 \
  .venv/bin/uvicorn api.server.main:app --port 8000
```

`/info` → `modalities` показывает фактический набор. Видео требует
`DECISION_VISION_MODEL_ID` и декодирования кадров; локальный CLIP видео не
поддерживает.

## 6. `BACKEND_UNAVAILABLE`

```json
{"error": {"code": "BACKEND_UNAVAILABLE", "backend": "systemone", "status_code": null}}
```

Внешний сервис (`DECISION_BACKEND=http` или fallback каскада) не ответил в
бюджет вызова. Тело запроса (`state`, `images`) в ответ и лог не попадает.

```bash
# жив ли внешний сервис и тот ли URL
curl -fsS "$DECISION_BACKEND_URL/health"
# бюджет и повторы
echo "$DECISION_BACKEND_TIMEOUT $DECISION_BACKEND_RETRIES"
journalctl -u decision-api@api -n 50 --no-pager | grep -i unavailable
```

- `status_code: 4xx` — внешний сервис отверг запрос; повторов не будет, чините
  контракт/авторизацию.
- `status_code: 5xx` или `null` — сеть/таймаут/перегрузка; увеличьте
  `DECISION_BACKEND_TIMEOUT`, проверьте сеть.
- Каскад при недоступном fallback продолжает отвечать локально, а `/ready`
  показывает `degraded: true` — это не ошибка, но эскалации не работают.

## 7. `FEEDBACK_UNSUPPORTED`

`/v1/feedback` — не универсальный приём разметки, а транспорт для каскада.
Нужен один из режимов:

- `DECISION_BACKEND=cascade` + `DECISION_CASCADE_RISK_BUDGET` +
  `DECISION_RISK_FEW_SHOT_N >= 2` — few-shot переобучение риск-модели;
- `DECISION_BACKEND=cascade` + `DECISION_CASCADE_POLICY=benefit` — уточнение
  точности внешней модели `q`.

Проверьте `/info` → `accepts_feedback`. Обычный порог уверенности обратную связь
не принимает.

## 8. Каскад не стартует: нет `q`, риск-модели или URL

Это ошибки конфигурации на старте (процесс не поднимается), а не HTTP-коды:

| Сообщение | Что сделать |
| --- | --- |
| `DECISION_BACKEND=http requires DECISION_BACKEND_URL` | задать URL внешнего SystemOne |
| `DECISION_BACKEND=cascade requires DECISION_CASCADE_FALLBACK_URL` | задать fallback URL |
| `DECISION_CASCADE_POLICY=benefit requires DECISION_CASCADE_REMOTE_ACCURACY` | снять `q`: `python -m eval.competitors.fit_remote_accuracy ...`, положить JSON/число в переменную |
| `DECISION_CASCADE_RISK_BUDGET requires a fitted risk model at ...` | обучить `risk.json` (`python -m eval.risk.run`) или указать `DECISION_RISK_MODEL_PATH` |
| `DECISION_UNTRAINED must be one of (...)` | `allow`, `refuse`, `fallback` или `auto` |

## 9. `/ready` отвечает `503`

Тело содержит `reasons`:

| Причина | Значение | Что делать |
| --- | --- | --- |
| `encoder is not loaded` | энкодер не загрузился | смотреть лог старта, `DECISION_MODEL_ID`, память/диск |
| `vision encoder is not loaded` | CLIP не загрузился | проверить `DECISION_VISION_MODEL_ID` и кэш HF |
| `no trained checkpoints for untrained_policy=refuse` | политика требует пак, паков нет | положить пак или `DECISION_UNTRAINED=allow` |
| `checkpoints are not readable by this service` | паки есть, но ни один не подошёл | см. §10 (dim/choice_head/подпись) |

`heads_loaded` и `checkpoints` в ответе показывают, сколько паков видит сервис и
сколько реально загружено.

## 10. Пак не подхватился: `dim` или `choice_head`

В логе прогрева:

```
WARNING skip checkpoint heads.pt: choice_head=option_scoring, service choice_head=linear
WARNING checkpoint dim=768 does not match features dim=64; ignoring ...
```

- `DECISION_FEATURE_DIM` должен совпадать с `feature_dim` из `metadata.json`
  пака (gpt2 — 768, hash-энкодер — 64).
- Режим голов задаётся `DECISION_CHOICE_HEAD` (`linear` или `option_scoring`) и
  должен совпадать с тем, как пак обучен. Переобучить под другой режим:
  `python -m training.recipe.train_heads --choice-head ...`.
- Подпись вопросов (`questions_signature`) должна совпадать: сверьте `questions`
  в `metadata.json` с запросом.

## 11. Первый запрос медленный

По умолчанию паки прогреваются на старте (`DECISION_WARMUP=1`). Если прогрев
выключен (`DECISION_WARMUP=0`), первый запрос платит за чтение чекпоинтов и
загрузку энкодера — это ожидаемо, но `/ready` в этом режиме не гарантирует, что
головы уже в памяти.

## 12. Порт занят

```
ERROR: [Errno 98] Address already in use
```

```bash
ss -ltnp | grep 8080
systemctl status decision-api@api.service
```

Один инстанс на порт; для нескольких — `decision-api@<instance>.service` с
разными портами в юните.
