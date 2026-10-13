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

## Позиционирование

### Хостед Jev против локального System One

| Критерий | Jev (платный SaaS) | System One Decision Model |
|----------|--------------------|---------------------------|
| Цена | Подписка/кредиты API | Бесплатно, открытый исходный код |
| Где работают данные | Облако вендора | Локально: ваш сервер, ваши данные |
| Модель | Закрытая, фиксированная | Выбирается вами: от gpt2 (CPU) до gemma-4-12B (GPU) |
| Выходной контракт | API вендора | Ваш: типизированный SDK, версиируемая схема |
| Приватность | Обращения уходят вовне | Ничего не покидает машину (после дистанционной загрузки весов) |
| Vendor lock-in | Есть | Нет: деплой через простые shell/systemd |
| Калибровка | Чёрный ящик | Встроенные метрики ECE/Brier + CLI |
| Обучение под домен | Нет | SFT + RLCD прямо в репозитории |
| Задержка | Сетевая (RTT + инференс вендора) | Локальная, настраиваемая (обрезка входа `DECISION_MAX_LENGTH`); см. «Оценка» |
| Внедрение | Только через их API | HTTP + Python SDK в вашем стеке |

Латентность измеряется прямо в репозитории (`eval/latency`) и сильно зависит
от модели, железа и длины входа: ограничивайте длину входа через
`DECISION_MAX_LENGTH` и закладывайте время на первый (холодный) инференс.

Честное ограничение: это **проект-прототип для контроля и прозрачности**,
а не готовый SaaS с UI. Если вам нужна «кнопка под ключ» — Jev удобнее; если нужны
контроль, приватность, цена и калибруемость — этот проект.

### Открытые альтернативы и совместимость

Тот же класс задач решают открытые проекты: **Cloudflare Clef** и
**Clef-Flash** (открытые веса для Workers AI / SGLang / vLLM),
**Perplexity `pplx-decider-v1-27b`** и **ConvAI Laya**. У них общий де-факто
протокол `POST /v1/systemone`: `model`, `state`, `questions` с
`type`/`instructions`/`criteria` и необязательными `images`/`videos`, ответ —
`answers` плюс `usage`.

Этот проект **говорит на том же протоколе** (`/v1/systemone`) и умеет
выступать шлюзом к ним (`DECISION_BACKEND=http`). Их не нужно выбирать
«вместо»: можно держать один локальный слой решений и при необходимости
проксировать к внешней модели.

### Что здесь уникально

- **Полный цикл в одном репозитории**: обучение (SFT) → RLCD-калибровка →
  метрики (ECE/Brier) → latency-профилирование → деплой.
- **Контракт-first**: типизированные результаты с версией схемы, а не свободный
  текст; клиент проверяет сумму распределения и `SCHEMA_VERSION`.
- **Честная мультимодальность**: сервис объявляет поддерживаемые модальности и
  отклоняет остальные типизированной ошибкой вместо тихой деградации.
- **Прозрачная задержка**: пофазный замер `encode`/`heads`/`decide` из коробки.
- **Протокол + локальное исполнение**: совместимость с экосистемой
  Jev/SystemOne без облачной зависимости.

## Установка окружения

```bash
bash scripts/bootstrap-user.sh        # uv, pyenv, venv, зависимости, gh

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# или через uv (быстрее):
uv pip install --python .venv/bin/python -r requirements.txt
```

`bootstrap-user.sh` работает без `sudo`: ставит `uv`/`pyenv` в `~/.local` и
`~/.pyenv`, создаёт `.venv`, ставит зависимости и кладёт `gh` в `~/.local/bin`.

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

### Вариант 3 — большая модель (gemma-4-12B, GPU)

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
| `DECISION_MODEL_ID` | имя модели на HF Hub | — (hash-энкодер) |
| `DECISION_FEATURE_DIM` | размерность признаков | `768` |
| `DECISION_MAX_LENGTH` | обрезка входа энкодера, токенов (`none` — без обрезки) | `256` |
| `DECISION_ENCODE_BATCH` | размер батча кодирования состояний (1–4096) | `16` |
| `DECISION_ENCODER_KIND` | вид текстового энкодера: `causal` (LM) или `encoder` (BERT-like) | `causal` |
| `DECISION_DEVICE` | устройство энкодера: `auto`, `cpu`, `cuda`, `cuda:N`, `mps` | `auto` |
| `DECISION_DTYPE` | тип весов энкодера: `float32`, `float16`, `bfloat16` | `float32` |
| `DECISION_CHECKPOINT_DIR` | каталог чекпоинтов голов | `checkpoints` |
| `DECISION_BACKEND` | исполнитель решений: `local`, `http` или `cascade` | `local` |
| `DECISION_BACKEND_URL` | базовый URL внешнего SystemOne-сервиса | — |
| `DECISION_BACKEND_MODEL` | имя внешней модели (поле `model`) | `remote` |
| `DECISION_BACKEND_MODALITIES` | модальности внешнего бэкенда | `text` |
| `DECISION_CASCADE_FALLBACK_URL` | URL fallback для `cascade` | — (обязателен) |
| `DECISION_CASCADE_FALLBACK_MODEL` | имя fallback-модели | `remote` |
| `DECISION_CASCADE_FALLBACK_MODALITIES` | модальности fallback | `text` |
| `DECISION_CASCADE_THRESHOLD` | порог уверенности для эскалации (0–1) | `0.6` |
| `DECISION_VISION_MODEL_ID` | CLIP/SigLIP для локальных `images` | — (без vision) |
| `DECISION_VISION_DIM` | размерность признаков vision-энкодера (1–8192) | `512` |
| `DECISION_MEDIA_ROOT` | allow-list каталог для чтения `images`/`videos` с диска | — (любой путь) |

Фактическое устройство и тип можно задать при запуске:

```bash
DECISION_MODEL_ID=openai-community/gpt2 \
DECISION_DEVICE=cuda DECISION_DTYPE=bfloat16 \
  .venv/bin/python -m uvicorn api.server.main:app --port 8000
```

`auto` выбирает CUDA при её наличии, иначе CPU. Недоступное устройство — явная
ошибка (без тихого отката), а `float16` на CPU запрещён. Наружу энкодер всегда
отдаёт CPU-тензоры, поэтому головы и кэш признаков работают одинаково на любом
устройстве.

По умолчанию текстовый энкодер — causal LM (`GemmaStateEncoder`,
`AutoModelForCausalLM`). Для двунаправленных encoder-only моделей (BERT-like)
задайте `DECISION_ENCODER_KIND=encoder` — тогда используется
`TransformerStateEncoder` (`AutoModel`) с тем же masked mean-pooling:

```bash
DECISION_MODEL_ID=bert-base-uncased DECISION_ENCODER_KIND=encoder \
DECISION_DEVICE=cuda \
  .venv/bin/python -m uvicorn api.server.main:app --port 8000
```

### Локальная мультимодальность

Локальный бэкенд умеет принимать `images`/`videos`, если у сервиса есть
vision-энкодер. Каждый энкодер сам объявляет свои модальности
(`VisionEncoder.modalities`), а сервис агрегирует их в `Capabilities`, поэтому
`/v1/systemone` отклонит ровно те модальности, что не поддержаны.

```bash
# CLIP включается переменной окружения (только изображения — видео требует
# отдельного декодирования кадров)
DECISION_VISION_MODEL_ID=openai/clip-vit-base-patch32 \
  uvicorn api.server.main:app --port 8000
```

Признаки состояния и медиа сливаются: вектор модальности усредняется и
проецируется в размерность текстовых признаков, после чего добавляется к
признакам каждого вопроса (вес — `media_kwargs.media_weight`, по умолчанию
`0.5`; `max_frames` управляет числом кадров видео). Для офлайн-тестов и
сервисов, где достаточно детерминированного сигнала, есть `HashVisionEncoder`.

```python
from api.server.service import DecisionService
from src.encoder.vision import HashVisionEncoder

svc = DecisionService(seed=0, vision_encoder=HashVisionEncoder(dim=64))
svc.modalities  # frozenset({'text', 'image', 'video'})
svc.decide("state", QUESTIONS, images=["shot.png"], videos=["clip.mp4"])
```

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

#### Каскад: локальные головы + внешняя модель (опционально)

Режим `DECISION_BACKEND=cascade` запускает **local-first** схему: сначала отвечают
локальные головы, а во внешнюю модель эскалируются только вопросы с
`confidence` ниже порога (или неподдерживаемую модальность — целиком). Так
внешний вызов платится лишь за действительно неясные случаи:

```bash
DECISION_BACKEND=cascade \
DECISION_CASCADE_FALLBACK_URL=http://127.0.0.1:30000 \
DECISION_CASCADE_FALLBACK_MODEL=clef-flash \
DECISION_CASCADE_THRESHOLD=0.6 \
  uvicorn api.server.main:app --port 8000
```

В `usage` возвращается `escalated` — число эскалированных вопросов, и `fallback`
с расходом внешнего бэкенда. Модальности каскада — объединение модальностей
primary и fallback.

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

Состояния кодируются батчами по `DECISION_ENCODE_BATCH` (по умолчанию 16), а не
всем набором сразу — на CPU это важно для памяти при обучении на больших наборах.

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

Есть три эндпоинта с одной и той же моделью решений:

- `POST /decisions` — собственный типизированный API (обёртка со
  `schema_version`), удобен для SDK и русскоязычных примеров;
- `POST /decisions/batch` — тот же контракт для списка состояний (`states`):
  все промпты кодируются одним вызовом (внутри — батчами по
  `DECISION_ENCODE_BATCH`), ответ — список `results`;
- `POST /v1/systemone` — стандартный протокол Jev/SystemOne, совместимый с
  Clef/Clef-Flash/Perplexity Decider (поле `model`, ответы в `answers`,
  `usage`, необязательные `images`/`videos`).

```bash
curl -s -X POST http://127.0.0.1:8000/decisions/batch \
  -H 'Content-Type: application/json' \
  -d '{
        "states": ["Stripe недоступен, теряем заказы", "Как сменить тариф?"],
        "questions": {
          "срочность": {"type": "noul",   "instructions": "Срочно?"},
          "команда":   {"type": "choice", "instructions": "Кто?", "options": ["биллинг", "интеграция"]}
        }
      }'
```

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

### Версионирование схемы (`schema_version`)

Версии — `<major>[.<minor>]`, совместимость — по **major** (см.
`src/versioning.py`): клиент, ожидающий `1.x`, принимает любой `1.y`, но не `2.*`.
**minor** — аддитивные изменения (необязательные поля/вопросы), **major** —
ломающие (удаление/переименование полей, смена семантики) с миграцией. Тот же
принцип применяется к чекпоинтам голов: payload содержит `schema_version`
(`CHECKPOINT_VERSION`), legacy-файлы без версии читаются как `1`, а более новый
major отклоняется явной ошибкой, а не молчаливым чтением неизвестного формата.

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

### Интеграция в чат-агента (SDK + каскад)

`api/sdk/agent.py` превращает типизированные решения в действие агента
(команда, приоритет, эскалация, тон ответа) чистой функцией `route_ticket` —
без сети, поэтому легко тестируется. Готовый пример:

```bash
# сервер должен быть поднят; при --fallback низкоуверенное обращение
# переспрашивает внешний SystemOne-сервис (клиентский каскад)
.venv/bin/python examples/chat_agent.py --fallback http://remote:8080
```

На стороне сервера каскад (`DECISION_BACKEND=cascade`) эскалирует низкоуверенные
решения сам — см. раздел «Каскад» выше; `route_ticket` может использоваться и
поверх него.

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

Замеры зависят от модели, железа и длины входа. Пример (8 vCPU, без GPU, gpt2,
вход обрезан до 256 токенов): первый запрос по новому состоянию ≈ 9 с;
повторные запросы с тем же состоянием — единицы-десятки миллисекунд за счёт
кэша признаков. На GPU и на коротких состояниях значения в разы ниже. Именно
отсутствие токен-генерации делает путь детерминированным и позволяет уложить
тёплый инференс в бюджет < 500 мс.

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

### Датасеты и офлайн-бенчмарк голов

Для воспроизводимой проверки качества есть генератор синтетических данных и
офлайн-бенчмарк — без сервера и внешних моделей.

```bash
# текст (тиккеты с метками) + изображения (фигуры) в ./data
.venv/bin/python -m eval.datasets.run --text-n 1000 --image-n 500 --out-dir data

# бенчмарк: обучение голов на train и метрики на отложенных val/test
.venv/bin/python -m eval.benchmark.run \
  --dataset data/support_tickets.jsonl --encoder hash --train-steps 200 \
  --val-fraction 0.2 --test-fraction 0.2 --out data/report.json
```

Генератор строит состояние из латентного сценария и вплетает в текст ключевые
слова, коррелирующие с меткой, — так головы действительно могут выучить
соответствие, а бенчмарк измерить его. Каждый вопрос кодируется своим промптом
(как в рантайме), поэтому головы не смешивают цели.

`--val-fraction`/`--test-fraction` включают честный train/val/test-сплит: головы
обучаются **только** на train, а метрики считаются на отложенных наборах
(`report["splits"]`), при этом поля `report["questions"]`/`overall_accuracy`
указывают на test. Без сплитов обучение и оценка идут на одних данных.

**Честная оговорка.** Синтетические метки линейно отделимы по ключевым словам,
поэтому даже hash-энкодер с линейной головой после 200 шагов SFT даёт
`overall_accuracy = 1.000` не только на train, но и на отложенных val/test
(по 200 примеров каждый). Это метрика **на синтетике**, и она доказывает
работоспособность конвейера, а не качество на реальных данных: без обучения та
же схема даёт ≈ 0.276 (уровень случайного угадывания). На реальном энкодере
та же задача решается уже неидеально: рецепт на gpt2 (малый прогон,
`--generate 60 --train-steps 60`) даёт train ≈ 0.69, val ≈ 0.47, test ≈ 0.51 —
то есть цифра `1.000` про hash-энкодер отражает лёгкость синтетики, а не
универсальное качество. Для `noul` дополнительно считаются Brier и ECE.
Датасеты и отчёты складываются в `data/` (в `.gitignore`).

### Публичная оценка (AG News): untrained vs trained

Кроме синтетики есть адаптер публичного датасета `fancyzhx/ag_news` (4 темы:
World/Sports/Business/Sci-Tech) и отчёт, который честно сравнивает необученные и
обученные головы на отложенном test и рисует reliability-диаграммы:

```bash
# hash-энкодер (офлайн, быстро)
.venv/bin/python -m eval.report.run --encoder hash --dim 64 --out-dir data/report

# реальный gpt2 на GPU
DECISION_DEVICE=cuda .venv/bin/python -m eval.report.run \
  --encoder openai-community/gpt2 --dim 768 --out-dir data/report
```

Отчёт кладёт `metrics.json` и `reliability_*.png` в `--out-dir`. Для `choice`/`score`
считаются top-1 accuracy, multi-class Brier и top-label ECE. Референс на AG News
(gpt2, GPU): overall accuracy 0.425 → 0.884, ECE по темам 0.725 → 0.151.

### Бенчмарк против конкурентов

`eval.competitors` сравнивает локальные головы, каскад и внешние
SystemOne-совместимые сервисы (Clef/Clef-Flash, Laya, Perplexity Decider) на одном
наборе и одном наборе вопросов: качество (accuracy/Brier/ECE), латентность
p50/p95/p99 и долю эскалаций.

```bash
# воспроизводимо и без сети (mock-конкурент)
.venv/bin/python -m eval.competitors.run --specs support --limit 200 \
  --mock --mock-latency-ms 5 --out data/competitors.json

# реальные сервисы
.venv/bin/python -m eval.competitors.run --specs public --limit 200 \
  --remote clef-flash=http://127.0.0.1:8000 --out data/competitors.json
```

Методика и зафиксированные числа — в `docs/competitor-benchmark.md`. Ключевой
вывод: каскад «local-first» сохраняет качество локальной модели, обращаясь к
внешнему сервису лишь за единицы процентов решений (0.911 против 0.908 у local при
1.94% эскалаций).

## Обучение голов

### SFT — учим на размеченных данных

```python
from src.decision_head.heads import DecisionHeads
from training.sft.train import SFTTrainer

heads = DecisionHeads(dim=768, questions={…})
# признаки подаются по-вопросно (как в рантайме): своя матрица на каждый вопрос,
# тогда каждая голова учится только на своих промптах и цели не смешиваются.
SFTTrainer(heads, lr=0.1).fit(
    features={"срочность": feats_noul, "команда": feats_choice, …},
    targets={"срочность": y_noul, "команда": y_choice, …},
    steps=80, batch_size=32,
)
# target: "choice"/"score" — классы-индексы, "noul" — float-цели в [0,1].
# Для одного вопроса допустим единый тензор features.
```

### Рецепт обучения голов (train/val/test)

Готовый рецепт обучает головы на train-сплите и сохраняет чекпоинт вместе с
метаданными (энкодер, размерность, хэш датасета, метрики по сплитам):

```bash
# сгенерировать данные и обучить головы
.venv/bin/python -m training.recipe.train_heads --generate 1000 \
  --out-dir checkpoints/support --encoder hash --dim 64 --train-steps 300

# или на своём датасете
.venv/bin/python -m training.recipe.train_heads \
  --dataset data/support_tickets.jsonl --out-dir checkpoints/support \
  --encoder openai-community/gpt2 --dim 768 --train-steps 300
```

На выходе `checkpoints/support/heads.pt` и `metadata.json`. Готовый end-to-end
рецепт (gpt2 + SFT + RLCD-гейт, train/val/test) — в
`training/recipe/train_reference_model.sh`: он сохраняет чекпоинт под **подписью
схемы вопросов**, поэтому сервер подхватывает его автоматически.

Матрица «несколько моделей × несколько датасетов» обучается одной командой
(`training/recipe/train_weight_packs.sh`); каждый пак — отдельный каталог
`checkpoints/packs/<dataset>-<model>/`, пригодный как `DECISION_CHECKPOINT_DIR`:

```bash
# {support, public} x {gpt2, distilgpt2, gpt2-medium}. Переопределяется окружением:
# SUPPORT_N/SUPPORT_STEPS/SUPPORT_RLCD, PUBLIC_LIMIT/PUBLIC_STEPS/PUBLIC_RLCD,
# PACKS_ROOT, DECISION_DEVICE, DECISION_DTYPE.
bash training/recipe/train_weight_packs.sh
```

Обученные веса весят мало и публикуются как **артефакт GitHub Release** (а не в
git) — либо один пак, либо вся матрица одним архивом + манифестом:

```bash
training/recipe/publish_release.sh v0.1.0 checkpoints/support       # один пак
training/recipe/publish_weight_packs.sh v0.1.0 checkpoints/packs    # матрица + manifest.json
```

RLCD-калибровка `noul`-голов на реальных признаках энкодера принимает изменение
только если оно улучшает калибровку на **отложенном** val (гейт «не навреди»):

```bash
.venv/bin/python -m training.recipe.calibrate_rlcd --specs public \
  --encoder openai-community/gpt2 --dim 768 --limit 4000 \
  --checkpoint-dir checkpoints/packs/public-gpt2 --device cuda
```

### Готовые паки весов (release)

Релиз [`v0.1.0`](https://github.com/slavabars/decision-cascade/releases/tag/v0.1.0)
содержит архив `v0.1.0-weights.tar.gz` (и `v0.1.0-manifest.json`) с шестью
обученными паками: `{support, public} × {gpt2, distilgpt2, gpt2-medium}`. Все числа
— на честном train/val/test (для `public` — настоящий test-сплит AG News):

| Пак | Энкодер | dim | Датасет | test acc |
|-----|---------|----:|---------|---------:|
| `support-gpt2` | gpt2 | 768 | синтетика (саппорт), n=1000 | **0.926** |
| `support-gpt2-medium` | gpt2-medium | 1024 | синтетика (саппорт), n=1000 | 0.914 |
| `support-distilgpt2` | distilgpt2 | 768 | синтетика (саппорт), n=1000 | 0.868 |
| `public-gpt2` | gpt2 | 768 | AG News, n=4000 | 0.880 |
| `public-distilgpt2` | distilgpt2 | 768 | AG News, n=4000 | 0.879 |
| `public-gpt2-medium` | gpt2-medium | 1024 | AG News, n=4000 | 0.873 |

```bash
tar -xzf v0.1.0-weights.tar.gz               # -> v0.1.0-weights/<pack>/
DECISION_CHECKPOINT_DIR=v0.1.0-weights/support-gpt2 \
DECISION_MODEL_ID=openai-community/gpt2 DECISION_FEATURE_DIM=768 \
  .venv/bin/python -m uvicorn api.server.main:app --port 8000
```

Паки `support-*` используют тот же набор вопросов, что сервис по умолчанию, и
подхватываются автоматически. Паки `public-*` соответствуют вопросам AG News
(`PUBLIC_QUESTIONS`) — их нужно подавать сервису вместе с этим набором вопросов.

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

В рецепте калибровка идёт через `calibrate_noul_heads`: веса применяются, только
если RLCD *улучшает* среднюю абсолютную ошибку (гейт «не навреди»), иначе голова
откатывается к состоянию после SFT. Ограничение: гейт оценивается на train-сплите.

## Тесты, линт, покрытие

```bash
.venv/bin/python -m pytest                          # 306 passed, 10 skipped
.venv/bin/python -m ruff check src api eval training tests examples scripts
.venv/bin/python -m pytest --cov=src --cov=api --cov=eval --cov=training
.venv/bin/python -m pytest -m gpu                   # GPU-тесты (auto-skip без CUDA)
shellcheck -x deploy/lib/common.sh deploy/ubuntu/install.sh deploy/centos/install.sh
```

Тесты на реальной модели — opt-in (по умолчанию пропускаются, чтобы `pytest` не
качал веса): `DECISION_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -m model` —
скачивает лёгкую `openai/clip-vit-base-patch32` (~0.6 ГБ) и проверяет
`ClipVisionEncoder` на реальных изображениях (форма, детерминизм, инвариантность к
батчу, проекция размерности, путь через `DecisionService`).

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
src/            — модель: encoder (+device.py), decision heads, sampler, бэкенды
                  (backends/), схемы (schemas.py)
api/            — FastAPI-сервер (server/) и Python SDK (sdk/: client.py, agent.py)
training/       — SFT (sft/), RLCD (rlcd/) и рецепт обучения голов (recipe/)
eval/           — калибровка (calibration/), latency (latency/), датасеты
                  (datasets/, вкл. публичный AG News), офлайн-бенчмарк (benchmark/),
                  отчёт untrained-vs-trained (report/) и сравнение с конкурентами
                  (competitors/)
deploy/         — Ubuntu/CentOS + systemd
docs/           — методика бенчмарка против конкурентов (competitor-benchmark.md)
examples/       — примеры (русский разбор обращения, чат-агент с каскадом)
tests/          — pytest-набор (306 тестов + opt-in GPU/модель)
data/           — сгенерированные датасеты и отчёты бенчмарков (в .gitignore)
scripts/        — bootstrap окружения (bootstrap-user.sh: uv/pyenv/venv/gh)
README.md       — этот файл
requirements.txt
pytest.ini
```

## Ограничения и планы

- **Качество по умолчанию**: головы по умолчанию случайные; осмысленные решения
  требуют SFT+RLCD на вашей разметке. Есть рецепты и обученные паки весов
  (`v0.1.0-weights.tar.gz` в GitHub Release: `support-*`/`public-*` × gpt2-family);
  в git веса не кладутся, поэтому после чистого клона «из коробки» — случайные
  головы, пока не скачаете пак или не обучите свои.
- **Устройство**: `DECISION_DEVICE`/`DECISION_DTYPE` выбирают CPU/GPU и тип;
  gemma-4-12B требует ~24 ГБ и CUDA, на CPU используйте gpt2.
- **Мультимодальность**: локальная CLIP поддерживает только изображения; видео
  требует отдельного декодирования кадров.
- **Честная оценка**: синтетические сплиты тривиально разделимы (`1.000` — это про
  конвейер, а не про домен). Публичные числа — на AG News (accuracy 0.425 → 0.88)
  и в бенчмарке против конкурентов; для своего домена нужна своя разметка.
- **Публикация весов**: релизом через `training/recipe/publish_release.sh` (один
  пак) или `training/recipe/publish_weight_packs.sh` (матрица).