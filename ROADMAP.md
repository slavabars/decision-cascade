# ROADMAP — от прототипа к конкурентоспособной платформе

Документ фиксирует проверенный статус проекта и план работ по закрытию оставшихся
пробелов (в первую очередь: GPU, обученные веса, сравнение с конкурентами).

> Статус на 2026‑10‑13. Проверено по коду и репозиторию, а не по описаниям.

---

## 1. Проверенный статус (что уже реализовано)

| Возможность | Статус | Где |
|---|---|---|
| Типизированные решения Noul/Choice/Score + валидация распределений | ✅ | `src/schemas.py`, `src/decision_head/heads.py` |
| Параллельная оценка всех вопросов одним прогоном | ✅ | `src/sampler/parallel.py` |
| Каскадный бэкенд local-first + эскалация по `confidence` | ✅ | `src/backends/cascade.py` |
| Батч-опрос многих состояний | ✅ | `POST /decisions/batch`, `DecisionService.decide_many` |
| Allowlist медиа `DECISION_MEDIA_ROOT` | ✅ | `src/encoder/vision.py`, `api/server/main.py` |
| Честная мультимодальность (`Capabilities` + `MODALITY_NOT_SUPPORTED`) | ✅ | `src/backends/*`, `api/server/main.py` |
| Единый бэкенд для всех эндпоинтов, синхронные эндпоинты, потокобезопасные кэши | ✅ (13.10) | `api/server/main.py`, `api/server/service.py` |
| `httpx` в зависимостях | ✅ (13.10) | `requirements.txt` |
| Masked mean pooling (padding не влияет на признаки) | ✅ (13.10) | `src/encoder/gemma.py`, `tests/test_encoder.py` |
| Выбор устройства/типа: `DECISION_DEVICE`/`DECISION_DTYPE` | ✅ (13.10) | `src/device.py`, `api/server/main.py` |
| GPU-путь протестирован (маркер `gpu`, CPU↔CUDA allclose) | ✅ (13.10) | `tests/test_gpu.py`, `pytest.ini` |
| Обученные веса «из коробки» (gpt2 + RLCD) | ✅ (13.10, релиз `v0.1.0`) | `training/recipe/train_weight_packs.sh`, `training/recipe/publish_weight_packs.sh` |
| Честная оценка на публичном датасете (AG News) | ✅ (13.10) | `eval/datasets/public_dataset.py`, `eval/report/run.py` |
| Калибровка для `choice`/`score` (Brier/ECE) | ✅ (13.10) | `eval/calibration/metrics.py`, `eval/benchmark/core.py` |
| Бенчмарк против конкурентов + mock | ✅ (13.10) | `eval/competitors/`, `docs/competitor-benchmark.md` |
| Тесты | ✅ **254 passing** (ruff + shellcheck чисто) | `tests/` |
| Протокол Jev/SystemOne + прокси к Clef/Laya/Perplexity | ✅ | `api/server/systemone.py`, `src/backends/http.py` |
| SFT + RLCD + рецепт train/val/test + публикация весов артефактом | ✅ (код) | `training/` |

## 2. Подтверждённые пробелы

> Все перечисленные пробелы закрыты 13.10 (см. §3 — статусы фаз). Таблица ниже
> сохранена как исходная фиксация проблем.

| Пробел | Факт проверки |
|---|---|
| **GPU реально не используется и не тестирован** | `api/server/main.py` жёстко задаёт `device="cpu"` и для `GemmaStateEncoder`, и для `ClipVisionEncoder`. В тестах нет ни одного упоминания `cuda/gpu/device`. GPU-ветка `GemmaStateEncoder` существует, но приложением не вызывается. |
| **Нет обученных весов «из коробки»** | Нет каталога `checkpoints/`, нет `*.pt` вне `.venv`, нет git-тегов и релизов. README честно предупреждает о случайных головах. |
| **Нет сравнения с конкурентами** | Есть только упоминания в README/докстрингах. Нет воспроизводимого прогона Clef/Clef-Flash/Laya/Perplexity в одном сценарии. |
| **Оценка только на синтетике** | `eval/datasets/` — генератор синтетических тикетов; метки линейно отделимы (`accuracy = 1.000`), реальных датасетов нет. |
| **Калибровка измеряется только для `noul`** | `eval/calibration/metrics.py` даёт ECE/Brier; для `choice`/`score` — только accuracy и `mean_expected_label`. |
| **Roadmap-фичи не сделаны** | Реальная калибровка на выходе gemma, трансформерный энкодер с attention-mask, интеграция в чат-агентов. |

### Дополнительно найденный дефект (важно) — ✅ исправлено (A1)

**Mean-pooling игнорирует `attention_mask`.** В `src/encoder/gemma.py`:
`hidden = ...hidden_states[-1]; pooled = hidden.mean(dim=1)`. При `padding=True`
усредняются и padding-позиции (pad = EOS). Значит признаки состояния **зависят от
соседей по батчу**: одно и то же состояние в разных батчах даёт разные векторы.
Это подрывает детерминизм, искажает кэш признаков (ключ — промпт, а значение
зависит от батча), обучение и все будущие замеры на реальном энкодере.
Исправлено masked mean pooling + тест инвариантности к составу батча.

---

## 3. План по фазам

Приоритеты: **A/B** — корректность и GPU (P0), **C** — веса (P0),
**D/E** — доказательства (P1), **F** — развитие (P2).
Оценки — в идеальных днях (S ≈ 0.5, M ≈ 1–2, L ≈ 3–5).

### Фаза A — Корректность энкодера (P0) · M — ✅ сделано

- [x] A1. Ввести masked mean pooling: `sum(hidden * mask) / sum(mask)` с учётом
  `attention_mask`; padding-позиции не влияют на вектор. Файл: `src/encoder/gemma.py`.
  Реализовано в `GemmaStateEncoder._mean_pool` (+ fallback на обычное среднее без маски).
- [x] A2. Инвариантный к батчу тест: один и тот же state в батче 1 и батче 8 даёт
  одинаковый вектор (`torch.allclose`), тест в `tests/test_encoder.py`.
- [x] A3. Зафиксировать поведение `max_length`/pad-token в докстринге и README.
  Докстринг обновлён; README — раздел «Устройство и тип».
- [x] A4. (опц.) Регрессионный тест на кэш: признаки не зависят от порядка/состава батча.

**Приёмка:** тест A2 зелёный; `pytest` не регрессирует; докстринг описывает pooling корректно.

### Фаза B — GPU-путь и тесты на GPU (P0) · M–L — ✅ сделано

- [x] B1. Переменные окружения: `DECISION_DEVICE` (`auto|cpu|cuda|cuda:N|mps`, default `auto`) и
  `DECISION_DTYPE` (`float32|float16|bfloat16`, default `float32`); валидация. Файл: `src/device.py`.
- [x] B2. Проброс в `create_service` и `create_vision_encoder` вместо `device="cpu"`;
  `auto` → `cuda` при `torch.cuda.is_available()`.
- [x] B3. Тесты-юниты на парсинг `DEVICE`/`DTYPE` (mock `torch.cuda.is_available`) — `tests/test_device.py`.
- [x] B4. GPU-интеграция под маркером `@pytest.mark.gpu` + `skipif not cuda`:
  крошечный gpt2 на CUDA — совпадение выхода с CPU (`allclose`), форма, fp16 — `tests/test_gpu.py`.
- [x] B5. `pytest.ini`: маркер `gpu` зарегистрирован (auto-skip без CUDA).
- [x] B6. `eval/latency/run.py --device --dtype --encoder`.
- [x] B7. README: таблица «CPU vs GPU» / раздел «Устройство и тип».

**Приёмка:** на CUDA-хосте сервер стартует с `DECISION_DEVICE=cuda`, выход совпадает
с CPU в допуске; без CUDA тесты не падают (auto-skip).

### Фаза C — Обученные веса «из коробки» (P0) · L — ✅ сделано (опубликовано v0.1.0)

- [x] C1. Референсная задача: тикет-саппорт (`text_dataset.QUESTIONS`) + публичный
  AG News для честной оценки (см. D). gpt2 на русском → нетривиальные метрики.
- [x] C2. Адаптер `eval/datasets/public_dataset.py` (AG News → noul/choice/score).
- [x] C3. Обучение референс-модели `training/recipe/train_reference_model.sh`
  (`--encoder openai-community/gpt2 --dim 768 --train-steps 500`).
- [x] C4. Калибровка `noul` через RLCD + **гейт «не навреди»**: если RLCD ухудшает
  калибровку, веса головы откатываются (`training/rlcd/train.py:calibrate_noul_heads`).
  Замерено: RLCD без гейта роняет noul-acc 1.00→0.53; с гейтом — без регрессии.
- [x] C5. Артефакт: `heads.pt` + копия `<questions_signature>.pt` + `metadata.json`
  (encoder, dim, max_length, sha256, specs/dataset_name, метрики train/val/test).
  Публикация — `publish_release.sh` (один пак) или `publish_weight_packs.sh`
  (матрица + `manifest.json`). **Опубликовано:** релиз
  [`v0.1.0`](https://github.com/slavabars/decision-cascade/releases/tag/v0.1.0).
- [x] C6. `training/recipe/train_reference_model.sh` (drop-in через `DECISION_CHECKPOINT_DIR`).
- [x] C7. Матрица паков `{support, public} × {gpt2, distilgpt2, gpt2-medium}` —
  `training/recipe/train_weight_packs.sh` (обучено на GPU). `train_heads.py` получил
  `--specs public` (AG News с настоящим test-сплитом). README-раздел — «Готовые паки весов».

**Замер (test-accuracy):** `support-gpt2` **0.926**, `support-gpt2-medium` 0.914,
`support-distilgpt2` 0.868; `public-gpt2` 0.880, `public-distilgpt2` 0.879,
`public-gpt2-medium` 0.873. noul-головы откалиброваны (ECE ≈ 0.01–0.02);
необученный сервис даёт `срочность ≈ 0.50`, обученный — `0.007` (неслучайно).

### Фаза D — Реальная оценка и калибровка (P1) · M–L — ✅ сделано

- [x] D1. `eval/calibration/metrics.py`: `multiclass_brier_score`, `top_label_ece`;
  `categorical_metrics` теперь даёт Brier/ECE для `choice`/`score`.
- [x] D2. Честный отчёт `untrained vs trained` — `eval/report/run.py` на AG News.
- [x] D3. Reliability-диаграммы для всех вопросов (`reliability_*.png`) + `metrics.json`.
- [x] D4. README «как воспроизвести» — раздел «Публичная оценка (AG News)».

**Замер (AG News, gpt2, HP=GPU):** overall accuracy **0.425 → 0.884**;
calibration (ECE): topic 0.725→0.151, business 0.122→0.063, tech_level 0.795→0.091.
Числа < 1.0, воспроизводимы одной командой.

### Фаза E — Воспроизводимый бенчмарк против конкурентов (P1) · L — ✅ сделано

- [x] E1. `eval/competitors/`: клиент `/v1/systemone` + адаптеры `local`, `cascade`,
  `http`-внешний (переиспользуются бэкенды проекта).
- [x] E2. Метрики: latency p50/p95/p99, accuracy/Brier/ECE по вопросам, escalation rate,
  агрегированный `usage` (стоимость).
- [x] E3. `MockCompetitor` на `httpx.MockTransport` (`eval/competitors/mocks.py`).
- [x] E4. CLI `python -m eval.competitors.run --target … --out report.json`.
- [x] E5. `docs/competitor-benchmark.md`: железо, версии, ключевой вывод.

**Замер:** `cascade[local->mock]` acc **0.911** против local **0.908** при **1.94%**
эскалаций — то же/лучшее качество при почти нулевых внешних вызовах.

### Фаза F — Развитие (P2)

- [x] F1. Трансформерный энкодер с attention-mask: `TransformerStateEncoder` (encoder-only,
  `AutoModel`) + общий `masked_mean_pool`; вид выбирается `DECISION_ENCODER_KIND`
  (`causal`/`encoder`) через `src/encoder/factory.py`. Тесты: unit + opt-in BERT (`-m model`).
- [x] F2. Реальная калибровка на выходе энкодера (RLCD в масштабе): гейт «не навреди»
  считается на **отложенном** val (`calibrate_noul_heads(gate_features=...)`),
  `train_benchmark(rlcd_gate_examples=...)` прокидывается из рецепта, скрипт
  `training/recipe/calibrate_rlcd.py` калибрует существующий пак и печатает
  Brier/ECE до/после. Тесты: гейт на held-out, проброс gate, smoke-скрипт.
- [x] F3. Пример интеграции в чат-агента (SDK + каскад): `api/sdk/agent.py`
  (`route_ticket` → команда/приоритет/эскалация/тон), пример
  `examples/chat_agent.py` с клиентским фолбэком на внешний SystemOne-сервис,
  тесты `tests/test_agent.py`.
- [ ] F4. Политика версионирования схемы (`SCHEMA_VERSION`) и миграций.

---

## 4. Порядок и зависимости

```
A (masked pooling) ─┬─► B (GPU) ─────────────┐
                    └─► C (веса) ──► D (реал. оценка) ──► E (конкуренты)
                                                   └─► F (развитие)
```

- A обязательна до любых замеров на реальном энкодере (иначе метрики недостоверны).
- B и C можно вести параллельно (разные роли).
- E опирается на D (нужен единый размеченный набор).
- F1 зависит от A.

## 5. Роли (из `agents.md`)

| Фаза | Роль |
|---|---|
| A, C, D | ml-engineer |
| B | ml-engineer + sysadmin-bridge (CUDA-драйверы — только через sudo-request) |
| E | evaluator + api-builder |
| F | architect |

## 6. Риски

- **Нет GPU-хоста** → закрыто: доступен NVIDIA RTX 5060 Ti (`torch 2.14.1+cu130`),
  GPU-тесты и обучение прогнаны.
- **Внешние API конкурентов** (ключи, приватность) → E опирается на `MockCompetitor`
  + открытые веса (Clef-Flash через vLLM/SGLang) локально; реальные прогоны — за
  пользователем (нужны URL/ключи).
- **Приватность весов**: не коммитим `*.pt` в git (уже в `.gitignore`) — только релиз
  (`training/recipe/publish_release.sh` / `publish_weight_packs.sh`; `gh` установлен
  и авторизован, релиз `v0.1.0` опубликован).

## 7. Definition of Done (общая)

- `ruff check` и `shellcheck` чисто; `pytest` зелёный (254 passed, 7 skipped —
  opt-in GPU/модель; GPU-тесты auto-skip без CUDA).
- Каждая фаза сопровождается тестами; README/ROADMAP обновляются по ходу.
- Никаких «сырых» чисел без команды воспроизведения и указания железа/версий.
