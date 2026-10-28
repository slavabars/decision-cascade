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
`--remote`. Релизы Clef-Flash и Laya такого HTTP-фасада не отдают, поэтому для
них есть `eval.competitors.release` (см. раздел с реальными прогонами ниже).

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

## Зафиксированный прогон: реальные релизы Clef-Flash и Laya

Ниже — замеры на **скачанных релизах**, а не на mock и не на «живом»
самописном сервисе. Оба конкурента подняты локально и отвечают на
`POST /v1/systemone` через `eval.competitors.release` (тонкий фасад: релизы
зовут `criteria`, а `score`-вероятности отдают по индексам уровней).

Окружение: RTX 5060 Ti 16 ГБ (driver 595.91.07), Python 3.11.17,
`torch 2.14.1+cu130`, `transformers 5.19.0`, `llama.cpp b11430`,
8 vCPU (AVX-512 BF16), 91 ГБ RAM.

| Участник | Что именно запущено | Параметры |
| --- | --- | --- |
| `local` | наши головы над `openai-community/gpt2` | 124M, `checkpoints/reference`, cuda |
| `local (AG News SFT)` | те же головы, SFT на train AG News (4000) | 124M, `checkpoints/ag_news`, cuda |
| `laya` | `convaiinnovations/laya` 0.4.15, `Router` | 421M / 322M (RU → multilingual), bf16, cuda |
| `clef-flash` | `Cloudflare/clef-flash` в llama.cpp, GGUF Q4_K_M | 9.4B, голова Q8_0, полный offload на GPU |

```bash
# Clef-Flash: GGUF-релиз в llama-server (он сам отдаёт /v1/systemone)
llama-server -m clef-flash-Q4_K_M.gguf -ngl 99 -c 16384 -ub 2048 \
    --no-warmup --no-webui --port 8093
python -m eval.competitors.release --model clef-flash \
    --base-url http://127.0.0.1:8093 --port 8091

# Laya: Router из pip-пакета
python -m eval.competitors.release --model laya --device cuda --port 8092

# замер (набор и вопросы одни и те же для всех участников)
DECISION_MAX_LENGTH=256 python -m eval.competitors.run --specs support --limit 200 \
    --remote laya=http://127.0.0.1:8092 \
    --remote clef-flash=http://127.0.0.1:8091 \
    --encoder openai-community/gpt2 --dim 768 \
    --checkpoint-dir checkpoints/reference --device cuda \
    --out data/competitors_real.json

# локальные головы под новый домен (SFT на train AG News)
DECISION_DEVICE=cuda DECISION_MAX_LENGTH=256 \
python -m training.recipe.train_heads --specs public --limit 4000 \
    --out-dir checkpoints/ag_news --encoder openai-community/gpt2 --dim 768 \
    --train-steps 300 --seed 0
```

> **Важно:** таблицы этого раздела сняты **до** температурной калибровки — так
> выглядит «сырое» сравнение сразу после SFT. Accuracy от калибровки не
> меняется, а вот каскад в этих прогонах почти не эскалирует (0–2%), потому что
> головы переуверены. Актуальные числа с калибровкой — в разделе
> «Пак голов под задачу» и «Калибровка температуры» ниже.

### `--specs support` (RU-тикеты, 200, наши головы обучены на этом генераторе)

```
target                           acc    p50ms    p95ms    esc%
----------------------------------------------------------------
local                          0.915    45.14    46.18    0.00
laya                           0.500    13.21    14.31    0.00
clef-flash                     0.667   311.55   318.81    0.00
cascade[local->laya]           0.918    45.48    53.45    1.25
cascade[local->clef-flash]     0.919    46.98   143.08    1.25
```

По вопросам (`accuracy`): local — 1.000 на пяти из шести и 0.490 на
`удержание`; laya — `эскалация` 0.820, `причина` 0.630, `срочность` 0.615,
`команда` 0.390, `удержание` 0.340, `накал` 0.205 (шкала из 5 уровней
схлопывается к середине); clef-flash — `причина` 0.975, `эскалация` 0.855,
`срочность` 0.745, `удержание` 0.665, `накал` 0.445, `команда` 0.320.

### `--specs public` (AG News, 200 test)

```
target                           acc    p50ms    p95ms    esc%
----------------------------------------------------------------
local (untrained heads)        0.422     9.71    14.40    0.00
laya                           0.662    16.42    20.40    0.00
clef-flash                     0.855   137.07   162.58    0.00
local (AG News SFT)            0.857     9.82    14.17    0.00
cascade[local->clef-flash]     0.863     9.80    17.12    1.17
cascade[local->laya]           0.863     9.74    17.84    1.17
```

По вопросам: `topic` — laya 0.965, clef-flash 0.895, local-SFT 0.845;
`business` — 0.885 / 0.915 / 0.915; `tech_level` — 0.135 / 0.755 / 0.810.
`tech_level` у laya ниже случайного (0.333): score-голова держит почти всё
распределение на среднем уровне (ожидаемый балл 1.25 при шкале 0..2).

### `--specs routerbench` (200 test)

```
target                           acc    p50ms    p95ms    esc%
----------------------------------------------------------------
local (untrained heads)        0.365    22.48    24.23    0.00
laya                           0.312    28.68    40.80    0.00
clef-flash                     0.442   182.98   240.43    0.00
```

По вопросам (`escalate`/`route`/`benefit`): local 0.225/0.400/0.470,
laya 0.395/0.285/0.255, clef-flash 0.455/0.390/0.480. Доли классов в этом
сплите: `escalate=yes` 77.5%, `route` cheap/mid/strong 40/31/29%,
`benefit` none/some/large 41/12/47%. То есть на `escalate` **все** модели
хуже тривиального «всегда да» (0.775): метка RouterBench — контрфактическая
(«дешёвая модель не справится»), и по одному тексту промпта её не вывести.

### Пак голов под задачу (task-adapted, 200)

Zero-shot — не наш режим: головы `Noul/Choice/Score` не переносятся «из
коробки», их обучают на разметке домена (`training/recipe/train_heads.py`,
`--specs public|routerbench`). Ниже — те же три набора, но с паком под задачу,
после температурной калибровки (см. следующий раздел).

```bash
# 1) пак под задачу (test-сплит честный: train/val/test не пересекаются)
DECISION_DEVICE=cuda DECISION_MAX_LENGTH=256 \
python -m training.recipe.train_heads --specs routerbench --limit 4000 \
    --out-dir checkpoints/packs/routerbench-gpt2 --encoder openai-community/gpt2 \
    --dim 768 --train-steps 500 --rlcd-steps 200 --seed 0

# 2) температура логитов по NLL на val (точность не меняется)
DECISION_DEVICE=cuda DECISION_MAX_LENGTH=256 \
python -m training.recipe.calibrate_temperature --specs routerbench \
    --encoder openai-community/gpt2 --dim 768 --limit 4000 \
    --checkpoint-dir checkpoints/packs/routerbench-gpt2

# 3) замер
DECISION_MAX_LENGTH=256 python -m eval.competitors.run --specs routerbench \
    --limit 800 --remote clef-flash=http://127.0.0.1:8091 \
    --encoder openai-community/gpt2 --dim 768 \
    --checkpoint-dir checkpoints/packs/routerbench-gpt2 --device cuda \
    --out data/competitors_real/routerbench_pack_clef.json
```

Всё это целиком (обучение, калибровка, подъём обоих конкурентов, три набора и
три `--sweep`) воспроизводит один скрипт — он же использован для чисел ниже:

```bash
CLEF_GGUF=/path/clef-q4km.gguf \
LLAMA_BIN=/path/llama-server \
LAYA_PYTHONPATH=/path/laya_pkg \
  eval/competitors/reproduce_real.sh --specs support,public,routerbench --n 200
```

`--n` — целевой размер test; размер датасета скрипт пересчитывает сам
(`--limit` в `eval.competitors.run` — это размер датасета, а test = limit/4 у
`public`/`routerbench` и limit у `support`). Уже обученные паки не
переобучаются (`SKIP_TRAIN=1` — не трогать их вовсе).

```
support (наши головы обучены на этом генераторе)
local (checkpoints/reference)  0.915    46.53    48.36    0.00
clef-flash                     0.667   305.87   313.72    0.00
laya                           0.500    13.30    14.70    0.00
cascade[local->clef-flash]     0.949   143.37   153.53   11.17
cascade[local->laya]           0.945    52.65    55.36   11.17

public (AG News, пак public-gpt2)
local (pack public-gpt2)       0.873     9.67    13.71    0.00
clef-flash                     0.855   130.11   155.47    0.00
laya                           0.662    15.71    19.50    0.00
cascade[local->clef-flash]     0.885     9.36   105.02    7.33
cascade[local->laya]           0.885     9.39    21.79    7.33

routerbench (пак routerbench-gpt2)
local (pack routerbench-gpt2)  0.542    21.01    23.35    0.00
clef-flash                     0.442   169.58   225.28    0.00
laya                           0.312    27.17    39.30    0.00
cascade[local->clef-flash]     0.460   188.10   234.47   85.00
cascade[local->laya]           0.343    46.08    60.53   85.00
```

На честном test-сплите (n=1000, не тот же срез, что в таблицах выше):
`public-gpt2` — 0.880, `routerbench-gpt2` — 0.548.

По вопросам RouterBench (`escalate`/`route`/`benefit`): local 0.775/0.360/0.490,
clef-flash 0.455/0.390/0.480, laya 0.395/0.285/0.255. То есть весь перевес
local на этом наборе даёт `escalate`, где он попадает в тривиальную константу
(0.775), а конкуренты — ниже неё; на `route` и `benefit` local примерно на
уровне Clef-Flash и выше Laya. Это не «мы лучше понимаем задачу», а «мы не
теряем на метке, которую по тексту не вывести».

### Калибровка температуры

Без неё головы переуверены, и каскад бесполезен: softmax по насыщенным логитам
даёт `confidence ≈ 1.0`, порог 0.6 не срабатывает, эскалаций 0%. Поэтому пак
после обучения калибруется по NLL на val (`training/recipe/calibrate_temperature.py`).
Температура делит логиты перед softmax/sigmoid, **argmax не меняется**, то есть
accuracy остаётся прежней.

| Пак | Вопрос | T | ECE (test) до → после | Brier (test) до → после |
| --- | --- | ---: | --- | --- |
| `routerbench-gpt2` | `escalate` | 30.02 | 0.214 → **0.022** | 0.214 → 0.168 |
| `routerbench-gpt2` | `route` | 307.21 | 0.547 → **0.024** | 1.136 → 0.666 |
| `routerbench-gpt2` | `benefit` | 17.23 | 0.454 → **0.123** | 0.938 → 0.538 |
| `public-gpt2` | `topic` | 11.36 | 0.144 → **0.024** | 0.295 → 0.227 |
| `public-gpt2` | `business` | 1.00 (гейт) | 0.100 → 0.100 | 0.101 → 0.101 |
| `public-gpt2` | `tech_level` | 9.88 | 0.087 → **0.011** | 0.179 → 0.147 |
| `checkpoints/reference` | `удержание` | 11.36 | 0.342 → **0.168** | 0.758 → 0.500 |

`T` подбирается только на val; метрики в таблице — на test, чтобы видно было
отсутствие подгонки. Точность не изменилась ни на одном вопросе.

**Гейт «не навреди».** Перед сохранением температура проверяется на том же val:
применяется, только если ECE улучшился **и** Brier не ухудшился. Иначе
остаётся `T = 1.0`. Это важно для бинарного `noul`: NLL-оптимум у него
смягчает уже откалиброванную голову — Brier выигрывает, а ECE по вероятности
положительного класса растёт (у `business` было 0.100 → 0.111, у
`public-distilgpt2` 0.107 → 0.135). Такой случай теперь отклоняется: у всех
трёх public-паков `business` остался `T = 1.0`. У голов `support`-паков, кроме
`удержание`, подбор и так даёт `T = 1.0` — там точность на val 1.0 и
вероятности уже честные.

Температура **никогда не опускается ниже 1.0**: NLL не ограничена снизу при
идеальном разделении классов, и на маленьком val (200 примеров, acc=1.0)
оптимум уходит в `T → 0`, превращая случайное совпадение в уверенность 1.0.
Калибровка только смягчает — это безопасное направление для движка решений.

Эффект на каскад: до калибровки эскалаций было 0–2%, теперь механизм работает.

```
public, --sweep (порог → accuracy / доля эскалаций)
0.00  0.873 /  0.00%      0.60  0.885 /  7.33%
0.20  0.872 /  0.33%      0.70  0.888 / 11.17%
0.30  0.873 /  0.67%      0.80  0.890 / 16.00%   <- максимум
0.50  0.880 /  3.17%      0.90  0.883 / 25.17%
```

На `public` оптимум 0.890 при 16% эскалаций (+1.7 п.п. к локальной модели);
порог по умолчанию 0.6 даёт 0.885 при 7.3%. После гейта «не навреди»
`business` остался неоткалиброванным, поэтому уверенность там выше, а
эскалаций меньше, чем с принудительным `T = 5.87` — это честнее, хотя выигрыш
каскада скромнее.
На `routerbench` каскад **не помогает ни при каком пороге** (0.542 → 0.547 в
пределах шума при 33–40% эскалаций, дальше — хуже): удалённая модель на этой
задаче слабее локальной, и калибровка это честно показывает. Вывод
практический: порог эскалации надо подбирать `--sweep` на своём домене, а не
брать 0.6 по умолчанию.

### Что из этого следует

- **В своём домене** обученные локальные головы (124M) дают 0.915 против
  0.500 у Laya и 0.667 у Clef-Flash, при p50 45 мс против 13 и 312 мс.
  Это ожидаемо: `support` — синтетический генератор с ключевыми словами,
  тест in-distribution, и число проверяет конвейер, а не обобщение.
- **На новом домене с разметкой** (AG News SFT) 124M локальная модель
  сравнивается с 9.4B Clef-Flash (0.857 против 0.855) при p50 10 мс против
  137 мс. Laya на этом наборе — 0.662.
- **Zero-shot, без своей разметки**, сильнее всех Clef-Flash (0.855), затем
  Laya (0.662), а наши головы «из коробки» случайны (0.422) — это ровно то
  ограничение, которое заявлено в README. Поэтому `DECISION_UNTRAINED=auto`
  отказывает, если каталог чекпоинтов задан явно: сравнивать zero-shot чужие
  модели с нашими необученными головами бессмысленно в обе стороны.
- **С паком под задачу** (те же 124M gpt2, 500 шагов SFT + 200 RLCD) local
  выигрывает на всех трёх наборах: 0.915 / 0.873 / 0.542 против 0.667 / 0.855 /
  0.442 у Clef-Flash и 0.500 / 0.662 / 0.312 у Laya. Цена — своя разметка
  (4000 примеров) и ~15 минут обучения на одной GPU; Clef-Flash при этом
  остаётся универсальнее (не требует обучения под каждый новый домен).
- **Каскад** включается только после калибровки: у некалиброванных голов
  `confidence ≈ 1.0`, порог 0.6 не срабатывает и эскалаций 0%. С температурой
  он даёт +3.4 п.п. на `support` (0.949 при 11.2% эскалаций) и +1.7 п.п. на
  `public` (0.890 при 16.0% эскалаций, порог 0.8), но **не помогает** на
  `routerbench`, где удалённая модель слабее локальной. Порог эскалации —
  параметр домена, его подбирают `--sweep`, а не берут 0.6 по умолчанию.
- **Латентность** сравнима только внутри одной таблицы (один набор, одна GPU):
  Clef-Flash — 4-битный GGUF (голова Q8_0), Laya и локальные головы — в полной
  точности. Laya режет вход до 512/1024 токенов, Clef-Flash читает до 16384,
  наши головы — 256 (`DECISION_MAX_LENGTH`), поэтому на длинных промптах
  RouterBench модели видят разный объём входа.
- **Публичные таблицы конкурентов несопоставимы с этими числами напрямую**:
  в карточке Clef-Flash RouterBench указан как «selected quality» (79.9), а
  здесь измеряется accuracy трёх типизированных вопросов (0.442); разные
  метрики, разные вопросы, разные сплиты.

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
- Замеры Clef-Flash в разделе с реальными прогонами сняты с **4-битного GGUF**
  (Q4_K_M, голова Q8_0): полный bf16-чекпоинт — 18.8 ГБ и в 16 ГБ VRAM не
  влезает. Разница «bf16 против Q4» здесь не измерялась, поэтому числа
  Clef-Flash — нижняя оценка его качества.
- Laya и Clef-Flash — **zero-shot** модели: их можно дообучать (у Laya в
  карточке 0.362 → 0.766 на своём typed-decisions наборе), поэтому сравнение
  «наш SFT против их zero-shot» честно только как «что даёт своя разметка»,
  а не как «мы лучше модели».
- Температура подбирается на val из **того же** распределения, что и test
  (in-distribution). Переносимость температуры на другой домен здесь не
  измерялась: при смене домена её нужно переподбирать вместе с головами, иначе
  вероятности снова разъедутся.
