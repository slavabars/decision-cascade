# Changelog

Все заметные изменения этого проекта.
Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версии — [Semantic Versioning](https://semver.org/lang/ru/).

## [Unreleased]

### Added

- **Hard gate «принимать только лучшее решение»** (`eval/risk/core.py`,
  `gate_risk_vs_confidence`): вариант принимается, только если он не хуже
  baseline `confidence` по AURC, eAURC, coverage@5% risk и acc@0.7 (допуск
  0.0005 поедает только шум). Считается в `eval/risk/run.py` и в
  `eval/risk/features.py` — по in-dist, по LOO (новый вопрос) и по кривой
  few-shot (точка против baseline при том же числе примеров); вердикты печатаются
  в отчёт и пишутся в `features.json`/`metrics.json` → `gate`.
- **Сертификат гейта в `risk.json`** (`ErrorRiskModel.gate`) и продакшен-фолбэк:
  при `accepted: false` `RiskAwareCascadeBackend` безусловно переходит на порог
  уверенности (`DECISION_CASCADE_THRESHOLD`), причина эскалации в `usage`
  становится `risk_gate`, в лог пишется предупреждение, `/info → engine.risk_gate`
  показывает `rejected`. Без сертификата режим работает как прежде.
- Замеры гейта (доля проходов `production` против `confidence`): in-dist — 2/3
  (RouterBench/hash, 3 seeds), 1/1 (RouterBench/gpt2), 1/1 (AG News/gpt2);
  LOO 0-shot — 5/9, 0/3 и 3/3 соответственно. Кривая few-shot: на gpt2
  `escalate` проходит с `n ≥ 5`, `route` и `benefit` не проходят ни при каком
  `n` (включая «все»); на hash — 4–5 из 9 при любом `n`; на AG News — 3/3 при
  `n = 0` и 2/3 дальше (добавление своих примеров переобучает `business`).
- Вариант `confidence` в кривой `--shots`: однопризнаковая baseline-модель в
  той же точке кривой — с ней и сравнивается гейт.
- Флаг `--state` в `eval/risk/features.py` и варианты `state[C]` / `prod+state[C]`:
  замер, несёт ли **замороженное состояние энкодера** (пулированный вектор
  промпта) сигнал об ошибке сверх шести признаков `DecisionResult`. Сила L2 у
  вариантов с состоянием своя (`--state-C`), потому что 768 измерений на 400
  строках валидации переобучаются.
- Базовая линия `random` в `eval/risk/features.py` — случайный балл («без
  навыка»), его AURC ≈ базовая частота ошибок; отделяет «модель не работает» от
  «модель хуже случайной».
- Флаг `--shots` в `eval/risk/features.py`: кривая дообучения на новом вопросе —
  модель обучается на валидации остальных вопросов плюс `n` случайных строк
  валидации отложенного вопроса, тест — его тест. Точки вложены (`n=0` = LOO,
  `shots=None` = in-dist). Позволяет оценить, сколько размеченных примеров нового
  вопроса нужно, чтобы снять потерю переноса.

### Changed

- Кривая `--shots` считает и baseline `confidence` в каждой точке, поэтому
  `SHOT_COLUMNS`/агрегат группируются по `(variant, shots)` — иначе точки разных
  вариантов смешивались бы в одну строку.
- `eval/risk/features.py` считает `production` и варианты с состоянием при одном и
  том же `C` (`--C`), а для вариантов с состоянием берёт `--state-C`; это убирает
  сравнение, в котором выигрыш давала не новая информация, а более сильная
  регуляризация.

### Измерено и не подключено

- Сигнал из **замороженного** состояния энкодера: на новый вопрос он переносится
  только на AG News (AURC 0.0403 против 0.0487 у `production` при равном L2) и
  только при сильной регуляризации; на RouterBench (и gpt2, и hash) улучшается
  только упорядочивание внутри вопроса. Подробности и таблицы —
  `docs/risk-aware-cascade.md`.
- Явный **свободный член на вопрос** вместо суррогата `n_classes`: на новом
  вопросе хуже (AURC 0.4182 против 0.4040 на gpt2/RouterBench), на фиксированном
  наборе — в пределах шума. Замена не выигрывает, поэтому `n_classes` оставлен.
- **Кривая дообучения на новом вопросе** (`--shots 0 5 20 ...`): десятка
  примеров достаточно, чтобы закрыть львиную долю потери переноса. Без GPU, не
  подключено в сервис (опт-ин измерение).

## [0.6.0] — 2026-10-16

### Added

- Голова выбора option-scoring (`DECISION_CHOICE_HEAD=option_scoring`): оценивает
  вариант общим билинейным скорером по его текстовому эмбеддингу, поэтому число
  параметров не зависит от числа вариантов, а набор вариантов можно подменить на
  инференсе без переобучения. Добавлены `build_option_prompt`,
  `build_option_embeddings`, замер `eval/option_transfer/run.py` и честные числа
  в `docs/option-scoring.md` (перенос работает на близких вариантах, но не на
  произвольных новых словах и не при конкуренции со знакомыми вариантами).
  Режим по умолчанию — `linear`; старые чекпоинты читаются без изменений.
- Флаг `--choice-head` в `eval/report/run.py` и `training/recipe/train_heads.py`.
- Флаг `--loo` в `eval/risk/features.py`: leave-one-question-out проверка переноса
  риск-модели на **новый** вопрос. Показала, что выигрыш новых признаков — это
  подстройка под известные вопросы, а не новый сигнал о конкретном решении;
  разбор и оговорка — в `docs/risk-aware-cascade.md`.

### Changed

- `ErrorRiskModel.from_dict` определяет имена признаков по числу весов, если в
  файле нет `feature_names`: трёхвесовый `risk.json` без списка признаков больше
  не падает при загрузке, а читается как модель прежнего формата.
- Оценщик риска `P(ошибка)` (`src/risk.py`) получил три признака к прежним трём:
  `log_odds_gap` (`ln(p1/p2)` двух лучших исходов), `n_classes` (число исходов) и
  `noul_logit` (знаковый логит `noul`). На RouterBench при `--train-steps 500` это
  даёт AURC 0.2633 против 0.3116 у прежней модели на gpt2 (+5.8 п.п. принятой
  точности при 50% эскалации) и 0.2547 против 0.2625 на hash; на AG News AURC
  0.0357 против 0.0509. Это метрика по всем вопросам сразу; проверка `--loo`
  показала, что выигрыш — **межвопросное упорядочивание**: внутри вопроса и на
  новом вопросе `confidence` не хуже, а на AG News новые признаки вредят. Числа,
  оговорки и команда воспроизведения — в `docs/risk-aware-cascade.md`. Модели,
  обученные на трёх прежних признаках, загружаются и работают в прежней
  семантике: признаки берутся по именам (`LEGACY_FEATURE_NAMES`).
- Замер состава признаков вынесен в `eval/risk/features.py`; вариант `production`
  в нём побитово совпадает с `risk_model` из `eval.risk.run`.

## [0.5.0] — 2026-10-16

### Added

- Политика обслуживания необученных голов `DECISION_UNTRAINED`
  (`allow`/`refuse`/`fallback`/`auto`): не отдавать решения со случайно
  инициализированных голов. `refuse` отвечает `409 UNTRAINED_HEADS`, `fallback`
  эскалирует каскад во внешнюю модель.
- Поле `trained` в ответах `POST /decisions` и `POST /decisions/batch` и новый
  эндпоинт `GET /info` (бэкенд, модальности, устройство, статус голов).
- Risk-aware каскад: эскалация по оценке `P(ошибка)` (`src/risk.py`) с бюджетом
  ошибок `DECISION_CASCADE_RISK_BUDGET` вместо порога на сырой уверенности
  (`DECISION_RISK_MODEL_PATH`, по умолчанию `<checkpoint>/risk.json`); причина
  эскалации — `usage.reason = "risk"`.
- Selective-метрики `eval/calibration/selective.py` (risk–coverage, AURC/eAURC,
  покрытие при бюджете риска) и честное сравнение `eval/risk/run.py`
  (локально / порог / бюджет) с обучением оценщика риска
  (`training/recipe/risk.py`). Методика и честные числа — в
  `docs/risk-aware-cascade.md`.

## [0.4.0] — 2026-10-14

### Added

- Self-hosted «живой» конкурент `eval/competitors/serve.py`: локальная causal-LM
  под протоколом SystemOne, отвечает по правдоподобию вариантов
  (sequence log-likelihood), а не генерацией. Реальные числа зафиксированы в
  `docs/competitor-benchmark.md`.
- CI-регрессия конвейера бенчмарка (`tests/test_competitors_regression.py`) и
  офлайн-тесты сервиса на крошечной локальной модели
  (`tests/test_competitors_live.py`).

### Changed

- Vision-энкодер `ClipVisionEncoder` грузит бэкбон через
  `AutoModel`/`AutoImageProcessor`, поэтому поддерживает не только CLIP, но и
  **SigLIP** (`google/siglip-*`) по одному `model_id` (токенизатор не грузится).
- Бренд унифицирован на **decision-cascade**: README, NOTICE, метаданные пакета,
  заголовок HTTP-API и служебные описания. `SystemOne` остаётся названием
  внешнего протокола совместимости (Jev/TypeSafe AI); в README добавлен раздел
  «О названии».

## [0.3.0] — 2026-10-13

### Added

- Публичный decision-датасет **RouterBench**: адаптер
  `eval/datasets/routerbench.py` (пин ревизии Hugging Face, вопросы
  `escalate`/`route`/`benefit`), интеграция в отчёт untrained-vs-trained
  (`eval/report`) и в сравнение с конкурентами (`eval/competitors`), а также
  свип `--sweep` для кривой «качество/доля эскалаций».
- Гигиена релиза: лицензия Apache-2.0 (`LICENSE`, `NOTICE`), упаковка
  (`pyproject.toml`, дистрибутив `decision-cascade`, единый `src.__version__`),
  CI (GitHub Actions: ruff + shellcheck + pytest), этот `CHANGELOG.md` и
  `SECURITY.md`.
- `pandas` добавлен в зависимости (нужен адаптеру RouterBench).

### Changed

- README: раздел публичной оценки RouterBench; позиционирование уточнено до
  честной beta вместо «проекта-прототипа».

## [0.2.0] — 2026-10-13

### Added

- Каталог запускаемых примеров `examples/` по сценариям (базовые вызовы, каскад,
  чат-агент, мультимодальность, интеграции, эксплуатация) с `examples/README.md`.
- Пак весов `v0.2.0-weights.tar.gz` (6 паков: `support-*`/`public-*` ×
  gpt2-family).

### Changed

- README приведён к каталогу примеров и актуализирован под v0.2.0.

## [0.1.0] — 2026-10-13

### Added

- Ядро: энкодеры (`Hash`, transformer, CLIP vision), типизированные головы
  решений (Noul/Choice/Score), схемы с `SCHEMA_VERSION`, батч-инференс.
- FastAPI-сервер (`/decide`, `/v1/systemone`) и Python SDK/чат-агент.
- Бэкенды: локальные головы, HTTP-прокси (SystemOne), каскад.
- Обучение: SFT, RLCD-калибровка, рецепт обучения голов и публикации весов.
- Оценка: калибровка (ECE/Brier), latency, датасеты (AG News), сравнение с
  конкурентами, отчёт untrained-vs-trained.
- Деплой: Ubuntu/CentOS + systemd; bootstrap без `sudo`.
- Первый пак весов `v0.1.0-weights.tar.gz`.

[Unreleased]: https://github.com/slavabars/decision-cascade/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/slavabars/decision-cascade/releases/tag/v0.6.0
[0.4.0]: https://github.com/slavabars/decision-cascade/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/slavabars/decision-cascade/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/slavabars/decision-cascade/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/slavabars/decision-cascade/releases/tag/v0.1.0
