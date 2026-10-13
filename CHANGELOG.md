# Changelog

Все заметные изменения этого проекта.
Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версии — [Semantic Versioning](https://semver.org/lang/ru/).

## [Unreleased]

### Added

- Self-hosted «живой» конкурент `eval/competitors/serve.py`: локальная causal-LM
  под протоколом SystemOne, отвечает по правдоподобию вариантов
  (sequence log-likelihood), а не генерацией. Реальные числа зафиксированы в
  `docs/competitor-benchmark.md`.
- CI-регрессия конвейера бенчмарка (`tests/test_competitors_regression.py`) и
  офлайн-тесты сервиса на крошечной локальной модели
  (`tests/test_competitors_live.py`).

### Changed

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

[Unreleased]: https://github.com/slavabars/decision-cascade/compare/v0.2.0...HEAD
[0.3.0]: https://github.com/slavabars/decision-cascade/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/slavabars/decision-cascade/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/slavabars/decision-cascade/releases/tag/v0.1.0
