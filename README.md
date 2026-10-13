# System One Decision Model

Machine-native система принятия решений в духе **Jev (TypeSafe AI)**, но с
открытым исходным кодом, локальным исполнением и полным циклом
«обучение → калибровка → оценка → деплой».

Вместо того чтобы заставлять LLM генерировать текст ответа, модель **решает**:
state-состояние превращается в признаки, а головы решений возвращают
**типизированные, вероятностные и калибруемые** ответы — Noul (да/нет),
Choice (выбор из опций) и Score (оценка на шкале) — без единого токена генерации.

## Что это за подход

### Машинные решения, а не генерация текста

Обычный LLM-бот «думает» свободным текстом и выдаёт ответ естественным языком.
Такой ответ невозможно:

- проверять на корректность (нет машиночитаемого контракта);
- калибровать (вероятность в тексте не измерить);
- агрегировать (парсинг выводов хрупок).

System One принимает решения как **функция состояния → вероятность**:

```
state (строка-описание) ──► StateEncoder ──► features [batch, dim]
                                              │
                     ┌────────────────────────┴─────────────────────────┐
                     ▼                        ▼                         ▼
                 NoulHead                ChoiceHead                ScoreHead
                 (sigmoid)               (softmax)                 (softmax·mean)
                     ▼                        ▼                         ▼
              NoulResult (p)          ChoiceResult (профайл)     ScoreResult (балл)
```

Результат — строгий датакласс с `confidence`, валидируемым распределением
(сумма = 1.0 ± 1e-6) и версией схемы. Это «System 1»-мышление: быстрое,
дискретное, вероятностное решение по состоянию — без глубокого рассуждения.

### Почему это надёжнее

- **Типизированные выходы** — каждый вопрос имеет заявленный тип и схему
  (Noul/Choice/Score), клиент проверяет ответ по контракту (`SCHEMA_VERSION = "1"`).
- **Вероятности, а не слова** — модель обязана выдать распределение; его можно
  мерять на калибровку (Expected Calibration Error, Brier score) и править.
- **Параллельная оценка** — все вопросы обрабатываются одним прогоном: один
  forward энкодера + N голов. Низкая задержка и дешёвая инференция.
- **Без авторегрессии** — нет цикла генерации токенов, поэтому latency
  детерминированна и мала (на CPU см. раздел «Оценка»).
- **Обучаемые головы** — головы можно дообучать под свои домены: SFT на разметке
  и RLCD (reinforcement learning for calibrated decisions, GRPO-подобный) для
  прямой оптимизации калибровки.

## Чем это лучше «платного Jev»

| Критерий | Jev (платный SaaS) | System One Decision Model |
|----------|--------------------|---------------------------|
| Цена | Подписка/кредиты API | Бесплатно, открытый код (MIT-ориентированная структура) |
| Где работают данные | Облако вендора | Локально: ваш сервер, ваши данные |
| Модель | Закрытая, фиксированная | Выбирается вами: от gpt2 (CPU) до gemma-4-12B (GPU) |
| Выходной контракт | API вендора | Ваш: типизированный SDK, версиируемая схема |
| Приватность | Обращения уходят вовне | Ничего не покидает машину (после дистанционной загрузки весов) |
| Vendor lock-in | Есть | Нет: деплой через простые shell/systemd |
| Калибровка | Чёрный ящик | Встроенные метрики ECE/Brier + CLI |
| Обучение под домен | Нет | SFT + RLCD прямо в репозитории |
| Задержка | Сетевая (RTT + инференс вендора) | Локальная, < 500 мс (реальная p50 ≈ 130 мс на gpt2/CPU) |
| Внедрение | Только через их API | HTTP + Python SDK в вашем стеке |

Честное ограничение: это **проект-прототип для контроля и прозрачности**,
а не готовый SaaS с UI. Если вам нужна «кнопка под ключ» — Jev удобнее; если нужны
контроль, приватность, цена и калибруемость — этот проект.

## Установка окружения

```bash
bash scripts/bootstrap-user.sh        # системные утилиты (cc, cmake, git, shellcheck)

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# или через uv (быстрее):
uv pip install --python .venv/bin/python -r requirements.txt
```

> `scripts/bootstrap.sh` — генератор скелета проекта; повторно запускать не нужно.

## Запуск сервера

### Вариант 1 — без сети / для мгновенной проверки (hash-энкодер)

```bash
.venv/bin/python -m uvicorn api.server.main:app --host 127.0.0.1 --port 8000
```

Энкодер — детерминированный `HashStateEncoder`: состояние превращается в
64-размерный вектор по хэшам токенов. Ничего не скачивается, всё работает офлайн.

### Вариант 2 — лёгкая модель gpt2 (~0.5 ГБ, hidden 768, CPU)

```bash
DECISION_MODEL_ID=openai-community/gpt2 \
  .venv/bin/python -m uvicorn api.server.main:app --host 127.0.0.1 --port 8000
```

При первом старте веса скачаются в `~/.cache/huggingface` (заложите время на
медленных каналах). Останавливается сервер командой kill по порту — см. «Остановка».

### Вариант 3 — большая модель (gemma-4-12B, дефолт проекта, GPU)

```bash
DECISION_MODEL_ID=google/gemma-4-12B \
  .venv/bin/python -m uvicorn api.server.main:app --host 127.0.0.1 --port 8000
```

Требует ~24 ГБ и видеокарту с CUDA. Без явного `DECISION_MODEL_ID` сервер
стартует с hash-энкодером.

## Как менять модель

Выбор модели — **через переменные окружения** при старте сервера
(`api/server/main.py` → `create_service`):

| Переменная | Назначение | По умолчанию |
|------------|------------|--------------|
| `DECISION_MODEL_ID` | имя модели на HF Hub | `google/gemma-4-12B` |
| `DECISION_FEATURE_DIM` | размерность признаков | `768` |
| `DECISION_CHECKPOINT_DIR` | каталог чекпоинтов голов | `checkpoints` |
| `DECISION_BACKEND` | исполнитель решений: `local` или `http` | `local` |
| `DECISION_BACKEND_URL` | базовый URL внешнего SystemOne-сервиса | — |
| `DECISION_BACKEND_MODEL` | имя внешней модели (поле `model`) | `remote` |
| `DECISION_BACKEND_MODALITIES` | модальности внешнего бэкенда | `text` |

### Бэкенды: локальные головы или внешняя модель

`/v1/systemone` обслуживается «бэкендом». По умолчанию это локальные головы
(`local`, только текст). Если задать `DECISION_BACKEND=http` и
`DECISION_BACKEND_URL`, сервис становится прокси к любому внешнему
SystemOne-совместимому сервису — Cloudflare Clef / Clef-Flash (Workers AI,
SGLang, vLLM), Perplexity Decider и т.п.:

```bash
DECISION_BACKEND=http \
DECISION_BACKEND_URL=http://127.0.0.1:30000 \
DECISION_BACKEND_MODEL=clef-flash \
DECISION_BACKEND_MODALITIES=text,image,video \
  uvicorn api.server.main:app --port 8000
```

Бэкенд объявляет поддерживаемые модальности (`Capabilities`). Если запрос
содержит неподдерживаемую модальность (например, `images` на текстовом
локальном бэкенде), сервис отвечает структурированной ошибкой, не пытаясь
тихо деградировать:

```json
{"error": {"code": "MODALITY_NOT_SUPPORTED",
           "supported": ["text"], "requested": ["text", "image"]}}
```

Указывать размерность необязательно: головы строятся по фактической размерности
выхода энкодера (`features.shape[-1]`), а если она отличается от 768 — вход
проектируется автоматически.

```bash
# gpt2 (hidden 768) — без проекции
DECISION_MODEL_ID=openai-community/gpt2 uvicorn …

# gpt2-medium (hidden 1024) — автоматическая проекция в 768
DECISION_MODEL_ID=openai-community/gpt2-medium uvicorn …

# любая другая causal-модель
DECISION_MODEL_ID=microsoft/phi-2 uvicorn …
```

Программный способ (без HTTP):

```python
import torch
from src.encoder.gemma import GemmaStateEncoder
from api.server.service import DecisionService

encoder = GemmaStateEncoder(
    model_id="openai-community/gpt2", feature_dim=768,
    dtype=torch.float32, device="cpu",
).load()

svc = DecisionService(encoder=encoder, dim=encoder.feature_dim)
results = svc.decide("текст состояния", QUESTIONS)
```

## Остановка сервера

```bash
# по PID, слушающему порт 8000
kill $(ss -ltnp | grep ':8000 ' | grep -oP 'pid=\K[0-9]+' | head -1)

# или по имени процесса
pkill -f "uvicorn.*api.server.main"

# проверка, что порт освобождён
ss -ltn | grep ':8000 ' || echo "порт 8000 свободен"
```

## Использование: справочник API

Есть два эндпоинта с одной и той же моделью решений:

- `POST /decisions` — собственный типизированный API (обёртка со
  `schema_version`), удобен для SDK и русскоязычных примеров;
- `POST /v1/systemone` — стандартный протокол Jev/SystemOne, совместимый с
  Clef/Clef-Flash/Perplexity Decider (поле `model`, ответы в `answers`,
  `usage`, необязательные `images`/`videos`).

### Формат вопросов

| Тип | Использование | Запрос |
|-----|---------------|--------|
| `noul` | Бинарный вопрос «да/нет» (вероятность) | `{"type": "noul", "instructions": "…", "criteria": ["что значит «да»", "что значит «нет»"]}` |
| `choice` | Выбор одной из опций | `{"type": "choice", "instructions": "…", "options": ["…", "…"], "criteria": {"…": "описание опции"}}` |
| `score` | Оценка по шкале уровней | `{"type": "score", "instructions": "…", "levels": ["…", "…"], "criteria": ["описание уровня 0", "описание уровня 1"]}` |

`instructions` и `criteria` входят в промпт энкодера вместе с `state` — то есть
модель отвечает именно на заданный вопрос, а не просто классифицирует состояние.
`criteria` необязательны и нужны для более точного разграничения опций.

### Формат ответа

Каждый результат — типизированный датакласс:

- **NoulResult**: `p` в [0, 1] — вероятность «да»; `confidence = 2·|p − 0.5|`;
- **ChoiceResult**: выбранная опция (`argmax`) и распределение по всем опциям;
- **ScoreResult**: взвешенное среднее индексов уровней (`score`) и распределение.

Все распределения валидируются: сумма = 1.0 ± 1e-6, границы соблюдены, ключи —
ровно перечисленные опции/уровни. Внешний конверт ответа:

```json
{
  "schema_version": "1",
  "decisions": {
    "имя_вопроса": { "type": "…", "…": "…" }
  }
}
```

### Протокол Jev/SystemOne (`/v1/systemone`)

```bash
curl -s -X POST http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "local",
        "state": "Stripe failing for 3 days, losing sales",
        "questions": {
          "urgent": {"type": "noul", "instructions": "Is this urgent?"},
          "team":   {"type": "choice", "criteria": {"billing": "money", "technical": "errors"}},
          "risk":   {"type": "score",  "criteria": ["low", "medium", "high"]}
        }
      }'
```

Ответ:

```json
{
  "model": "local",
  "answers": {
    "urgent": {"type": "noul", "probability": 0.72, "confidence": 0.44},
    "team":   {"type": "choice", "choice": "technical",
               "confidence": 0.61, "probabilities": {"billing": 0.39, "technical": 0.61}},
    "risk":   {"type": "score", "score": 1.4, "confidence": 0.52,
               "probabilities": {"low": 0.2, "medium": 0.4, "high": 0.4},
               "legend": ["low", "medium", "high"]}
  },
  "usage": {"questions": 3}
}
```

Особенности протокола: `instructions` необязателен (при отсутствии берётся id
вопроса), а для `choice`/`score` допустимые варианты можно задавать только через
`criteria` (ключи/элементы становятся опциями). Поля `images`, `videos`,
`media_kwargs` принимаются; если бэкенд их не поддерживает — возвращается ошибка
`MODALITY_NOT_SUPPORTED`.

### curl

```bash
curl -s -X POST http://127.0.0.1:8000/decisions \
  -H 'Content-Type: application/json' \
  -d '{
        "state": "Stripe failing for 3 days, losing sales",
        "questions": {
          "urgent":  {"type": "noul",   "instructions": "Обращение срочное?"},
          "team":    {"type": "choice", "instructions": "Кто должен взять?",
                       "options": ["биллинг", "интеграция", "продажи"]},
          "anger":   {"type": "score",  "instructions": "Уровень недовольства",
                       "levels": ["спокоен", "недоволен", "крайне"]}
        }
      }'
```

### SDK (Python)

```python
from api.sdk.client import DecisionClient

with DecisionClient(base_url="http://localhost:8000") as client:
    result = client.decisions(
        "Уже третьи сутки не работают платежи, теряем заказы",
        {
            "срочность": {"type": "noul", "instructions": "Срочно?"},
            "причина":   {"type": "choice", "instructions": "Причина?",
                          "options": ["биллинг", "интеграция", "продажи"]},
        },
    )
# result — проверен по схеме типов и распределений
```

### Готовый русский пример

```bash
.venv/bin/python examples/client.py
```

Разбор обращения в техподдержку: шесть вопросов всех трёх типов одним запросом
с человекочитаемым выводом вероятностей и уверенности по каждому.

## Оценка качества

### Калибровка (ECE / Brier)

Вероятность «в среднем оправдывается»? Для этого есть `eval/calibration`:
ECE (взвешенное отклонение accuracy от confidence по бинам), Brier score,
reliability diagram.

```bash
.venv/bin/python eval/calibration/run.py \
  --probs probs.npy --outcomes outcomes.npy --bins 10
```

Классы NoulHead «калибруются» через RLCD; метрики выводятся как `metrics.json`
и `reliability.png` в `eval/calibration/out/`.

### Латентность (бюджет < 500 мс)

```python
import httpx
from eval.latency.benchmark import LatencyBenchmark

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=60) as c:
    print(LatencyBenchmark(
        callable=lambda: c.post("/decisions", json=PAYLOAD)).run(warmup=2, n=5))
```

Измерено на CPU с gpt2: **p50 ≈ 130 мс**, p95 ≈ 158 мс, p99 ≈ 164 мс —
комфортно ниже бюджета, т.к. нет токен-генерации.

Готовый CLI с разбивкой по фазам (`encode` / `heads` / `decide`):

```bash
# in-process, без HTTP и без внешней модели (hash-энкодер по умолчанию)
.venv/bin/python -m eval.latency.run --n 30 --warmup 3

# через запущенный сервер (свой /decisions или внешний SystemOne)
.venv/bin/python -m eval.latency.run --http http://127.0.0.1:8000/decisions --n 30
```

Фаза `encode` замеряется прямым прогоном энкодера (в обход кэша признаков), а
`decide` — полным путём сервиса. На повторяющемся `state` `decide` попадает в
кэш и потому быстрее `encode`; это наглядно показывает вклад кэша.

## Обучение голов

### SFT — учим на размеченных данных

```python
from src.decision_head.heads import DecisionHeads
from training.sft.train import SFTTrainer

heads = DecisionHeads(dim=768, questions={…})
# target: "choice"/"score" — классы-индексы, "noul" — float-цели в [0,1]
SFTTrainer(heads, lr=0.1).fit(features, targets, steps=80, batch_size=32)
```

### Чекпоинты голов и переиспользование

Головы решений создаются один раз на набор вопросов и переиспользуются между
запросами (кэш по подписи схемы). Обученные веса сохраняются и подгружаются:

```python
svc.save_heads(dim=768, questions=QUESTIONS)      # -> checkpoints/<schema>.pt
```

При старте сервер читает каталог из `DECISION_CHECKPOINT_DIR` (по умолчанию
`checkpoints/`) и, если для схемы есть чекпоинт с подходящей размерностью,
загружает веса. Это делает качество воспроизводимым между перезапусками, а
латентность — без повторного прогона энкодера по неизменному состоянию (кэш
признаков по промпту).

### RLCD — учим калибровке

RLCD (reinforcement learning for calibrated decisions): reward `−|p − outcome|`,
GRPO-группы из нескольких выборок, policy gradient. Прямо оптимизирует
правдоподобие вероятностей.

```python
from training.rlcd.train import RLCDNoulTrainer

trainer = RLCDNoulTrainer(dim=2, lr=0.2)
trainer.train(features, outcomes, steps=300)
```

## Тесты, линт, покрытие

```bash
.venv/bin/python -m pytest                          # 118 passed
.venv/bin/python -m ruff check src api eval training tests examples
.venv/bin/python -m pytest --cov=src --cov=api --cov=eval --cov=training
shellcheck -x deploy/lib/common.sh deploy/ubuntu/install.sh deploy/centos/install.sh
```

> Совет: на машинах с малым объёмом RAM полный `pytest --cov` в одном процессе
> может упасть (SIGILL на latency-тесте из-за кумулятивных torch-аллокаций).
> Обход: два прогона coverage с `--parallel-mode` и `coverage combine`.

## Deploy

```bash
sudo bash deploy/ubuntu/install.sh      # Ubuntu — apt, venv, systemd-юнит
sudo bash deploy/centos/install.sh      # CentOS/RHEL — dnf, venv, systemd-юнит
systemctl status decision-api@api.service
```

Обе сборки используют общий `deploy/lib/common.sh`: детект дистрибутива по
`/etc/os-release`, установка пакетов, `ensure_venv`, установка юнита
`decision-api@.service` (uvicorn, порт 8080, автозапуск).

## Структура репозитория

```
src/            — модель: encoder, decision heads, sampler, бэкенды (backends/), схемы (schemas.py)
api/            — FastAPI-сервер (server/) и Python SDK (sdk/client.py)
training/       — SFT (sft/) и RLCD (rlcd/)
eval/           — калибровка (calibration/) и latency (latency/)
deploy/         — Ubuntu/CentOS + systemd
examples/       — примеры (русский разбор обращения)
tests/          — pytest-набор (118 тестов)
scripts/        — скрипты окружения (bootstrap.sh — скелет, не для повторного запуска)
README.md       — этот файл
requirements.txt
pytest.ini
.pre-commit-config.yaml
.opencode/      — конфигурация opencode-агента (не публикуется)
```

## Ограничения и планы

- **Качество по умолчанию**: головы по умолчанию случайные; осмысленные решения
  требуют SFT+RLCD на вашей разметке. Обученные веса сохраняются в чекпоинты и
  подгружаются при старте, но готовой обученной модели «из коробки» нет — это
  особенность платформы, а не готовая модель.
- **GPU-модели** (gemma-4-12B) требуют CUDA-машины; на CPU используйте gpt2.
- **Roadmap**: реальная калибровка на gemma-выходе, батч-опросы многих состояний,
  вариант энкодера-трансформера с attention-mask состояний, интеграция в чат-агентов.