# Бенчмарк против конкурентов

Воспроизводимый протокол сравнения decision-cascade с внешними
SystemOne-совместимыми сервисами (Cloudflare Clef / Clef-Flash, Laya,
Perplexity Decider и любые vLLM/SGLang-эндпоинты на `POST /v1/systemone`).

Все участники измеряются на **одном размеченном наборе** и **одном наборе
вопросов**:

- `eval.competitors.run` — CLI;
- `eval.competitors.evaluate.evaluate_backend` — метрики одного бэкенда;
- `eval.competitors.mocks.mock_systemone_transport` — детерминированный
  mock-сервис на `httpx.MockTransport` (CI, без сети);
- `eval.competitors.serve` — self-hosted «живой» конкурент: локальная causal-LM
  под протоколом SystemOne.

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

## Self-hosted «живой» конкурент

Чтобы получить настоящие числа конкурента без внешних API, поднимите локальный
SystemOne-сервис поверх любой causal-LM (`eval.competitors.serve`). Он **не
генерирует** ответ, а оценивает правдоподобие каждого варианта
(`option`/`level` или `yes`/`no`) как продолжения подсказки (sequence
log-likelihood) и нормирует их softmax, поэтому вероятности пригодны для
Brier/ECE, а не только для accuracy.

```bash
# терминал 1 — сервис (веса скачаются в кэш HF при первом запуске)
python -m eval.competitors.serve --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --port 8090 --device cuda

# терминал 2 — бенчмарк против него
DECISION_MAX_LENGTH=256 python -m eval.competitors.run --specs support --limit 200 \
    --remote live=http://127.0.0.1:8090 \
    --encoder openai-community/gpt2 --dim 768 \
    --checkpoint-dir checkpoints/reference --device cuda \
    --out data/competitors_live.json
```

`--model` принимает любую HF causal-LM; `--remote` одинаково подключает и
настоящие развёртывания Clef/Clef-Flash/vLLM (у них тот же `POST /v1/systemone`),
поэтому этот сервис нужен лишь для воспроизводимого локального прогона.

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

## Зафиксированный прогон: живой конкурент

`--specs support --limit 200`, локальные головы — `checkpoints/reference` (gpt2),
конкуренты — self-hosted causal-LM (`eval.competitors.serve`), GPU RTX 5060 Ti:

```
target                         acc    p50ms    p95ms    esc%
------------------------------------------------------------
local                         0.915    49.79    52.16    0.00
mock                          0.378     2.32     2.48    0.00
live-gpt2 (base LM)           0.303   351.73   372.10    0.00
live-smol (135M instruct)     0.201   291.45   309.22    0.00
cascade[local->live-gpt2]     0.913    50.03   132.73    1.25
```

Англоязычный AG News (`--specs public --limit 800`; локальные головы не обучены
под эти вопросы, поэтому `local` — базовый уровень):

```
target                         acc    p50ms    p95ms    esc%
------------------------------------------------------------
local (untrained)             0.268     0.57     0.81    0.00
live-smol                     0.567    49.12    67.53    0.00
live-gpt2                     0.353    32.79    43.69    0.00
cascade[local->live-smol]     0.567    50.20    67.59  100.00
```

Вывод: конвейер измеряет настоящие модели, а не mock. Базовая gpt2 на домене
почти случайна (0.303); маленькая instruct-модель SmolLM2-135M слаба на русском
(0.201), но на английском AG News даёт 0.567. Каскад сохраняет качество локальных
голов (0.913 против 0.915) при 1.25% эскалаций. Для сравнения с Clef/Laya/
Perplexity подставьте их URL в `--remote` — команда и метрики те же.

## Оговорки

- Mock-конкурент **методологический**: он проверяет конвейер и повторяемость,
  но не качество реальных Clef/Laya/Perplexity. Для честного сравнения
  подключайте реальные эндпоинты через `--remote`.
- Self-hosted живой конкурент (`eval.competitors.serve`) отвечает по
  правдоподобию вариантов, а не генерацией; его числа зависят от выбранной
  модели и не заменяют замеры настоящих Clef/Laya/Perplexity.
- Латентность внешних сервисов зависит от сети и железа — фиксируйте их при
  публикации чисел.
- Ни один из публичных конкурентов (Clef-Flash, Laya, Perplexity Decider) не
  предоставляет каскад «из коробки»: экономия достигается тем, что основную
  часть решений принимает локальная модель.
