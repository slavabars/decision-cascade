# Бенчмарк против конкурентов

Воспроизводимый протокол сравнения System One Decision с внешними
SystemOne-совместимыми сервисами (Cloudflare Clef / Clef-Flash, Laya,
Perplexity Decider и любые vLLM/SGLang-эндпоинты на `POST /v1/systemone`).

Все участники измеряются на **одном размеченном наборе** и **одном наборе
вопросов**:

- `eval.competitors.run` — CLI;
- `eval.competitors.evaluate.evaluate_backend` — метрики одного бэкенда;
- `eval.competitors.mocks.mock_systemone_transport` — детерминированный
  mock-сервис на `httpx.MockTransport` (CI, без сети).

## Что измеряется

| Метрика | Смысл |
| --- | --- |
| `overall_accuracy` | средняя accuracy по всем вопросам |
| `questions[name]` | `accuracy`/`brier`/`ece` по каждому вопросу (noul и categorical) |
| `latency_ms` | p50/p95/p99 одного запроса `decide` (мс) |
| `escalation_rate` | доля решений-вопросов, ушедших во внешний сервис (для каскада) |
| `usage_totals` | расход из `usage` ответа (вложенные поля складываются) |

Для `noul` качество считается по вероятностям (`noul_metrics`), для
`choice`/`score` — по распределению (`categorical_metrics`: top-1 accuracy,
multi-class Brier, top-label ECE). Каскад, помимо качества, показывает
`requests_with_escalation` — сколько запросов потребовали внешнего вызова.

## Наборы (`--specs`)

| `--specs` | Источник | Вопросы |
| --- | --- | --- |
| `support` | синтетические тикеты (`eval.datasets.text_dataset`) | 6 вопросов всех типов |
| `public` | AG News (`eval.datasets.public_dataset`) | тема/бизнес/техноуровень |
| `routerbench` | RouterBench (`eval.datasets.routerbench`) | `escalate`/`route`/`benefit` |

`routerbench` — публичный decision-датасет маршрутизации LLM (см.
`docs/routerbench.md`): метки реальные, промпты разнородные. Для него доступен
`--shot 0shot|5shot` (по умолчанию `0shot`).

## Участники

- `local` — наши локальные головы (`LocalHeadsBackend` над `DecisionService`;
  опционально обученные веса через `--checkpoint-dir`).
- `<name>` — внешний сервис, заданный `--remote NAME=URL`.
- `mock` — детерминированный mock-конкурент (`--mock`).
- `cascade[local->X]` — `CascadeBackend`: локальные головы, при
  `confidence < threshold` вопрос эскалируется во внешний бэкенд `X`.

## Воспроизводимый прогон (без сети)

```bash
python -m eval.competitors.run --specs support --limit 200 \
    --mock --mock-latency-ms 5 --out data/competitors.json
```

С обученной референс-моделью (см. `training/recipe/train_reference_model.sh`):

```bash
DECISION_MAX_LENGTH=256 python -m eval.competitors.run --specs support --limit 200 \
    --mock --mock-latency-ms 2 \
    --encoder openai-community/gpt2 --dim 768 \
    --checkpoint-dir checkpoints/reference --device cuda \
    --out data/competitors_trained.json
```

## Реальные конкуренты

```bash
python -m eval.competitors.run --specs public --limit 200 \
    --remote clef-flash=http://127.0.0.1:8000 \
    --remote laya=https://api.laya.example \
    --remote perplexity=https://api.perplexity.example \
    --out data/competitors_real.json
```

Любой сервис, отвечающий на `POST /v1/systemone` схемой
`{"answers": {name: {...}}, "usage": {...}}`, подключается одной строкой
`--remote`.

## Кривая «качество против доли эскалаций» (`--sweep`)

Каскад — это компромисс: чем ниже порог уверенности, тем больше внешних вызовов
и тем выше качество. `--sweep` считает всю кривую (порог от 0.0 до 1.0) и кладёт
её в отчёт как `cascade_sweep`:

```bash
python -m eval.competitors.run --specs routerbench --limit 500 \
    --mock --mock-latency-ms 2 --sweep --sweep-points 11 \
    --out data/competitors_routerbench.json
```

Вывод: таблица `cascade sweep -> <fallback>` (`threshold`, `acc`, `esc%`, `p50ms`)
и тот же набор строк в JSON. Порог `0.0` — чисто локальный ответ, `1.0` — всё
уходит во внешний сервис; между ними виден фронт «качество/цена».

## Зафиксированный прогон (референс)

Окружение: Python 3.11.17, `torch 2.14.1+cu130`, NVIDIA GeForce RTX 5060 Ti,
`transformers 5.19.0`, `httpx 0.28.1`, `numpy 2.4.6`.

`--specs support --limit 60 --mock --mock-latency-ms 2`, локальные головы —
`checkpoints/reference` (gpt2):

```
target                           acc    p50ms    p95ms    esc%
----------------------------------------------------------------
local                          0.908    49.18    53.48    0.00
mock                           0.378     2.32     2.48    0.00
cascade[local->mock]           0.911    49.93    55.45    1.94
```

Вывод: каскад сохраняет качество локальной модели (`0.911` против `0.908`),
отправляя во внешний сервис лишь **1.94%** решений-вопросов. Это и есть
экономия: при сопоставимом (или лучшем) качестве внешний сервис вызывается
почти никогда.

## Оговорки

- Mock-конкурент **методологический**: он проверяет конвейер и повторяемость,
  но не качество реальных Clef/Laya/Perplexity. Для честного сравнения
  подключайте реальные эндпоинты через `--remote`.
- Латентность внешних сервисов зависит от сети и железа — фиксируйте их при
  публикации чисел.
- Ни один из публичных конкурентов (Clef-Flash, Laya, Perplexity Decider) не
  предоставляет каскад «из коробки»: экономия достигается тем, что основную
  часть решений принимает локальная модель.
