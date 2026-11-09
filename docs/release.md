# Релиз: версия, PyPI и артефакты

Процесс один для патча, минорной и мажорной версии; отличается только объёмом
CHANGELOG и решением «ломаем ли контракт» (ломать можно только в major).

## 1. Подготовка

1. **Версия** — `src/__init__.py` (единственный источник: `pyproject.toml`
   берёт её через `dynamic = ["version"]`).
2. **CHANGELOG** — раздел `## [X.Y.Z] — YYYY-MM-DD` и ссылка
   `[X.Y.Z]: .../compare/v<предыдущая>...vX.Y.Z`; `[Unreleased]` переключается
   на сравнение с новой версией.
3. **Документация** — если меняется поведение: README, `docs/*.md`,
   `docs/stability.md` (статус поверхности: стабильно или beta).
4. **Тесты** — `tests/test_release.py` проверяет, что версия, README, CHANGELOG
   и классификатор пакета согласованы. Расхождение — красный CI.

## 2. Проверки перед тегом

```bash
.venv/bin/python -m ruff check src api eval training tests examples scripts
shellcheck -x $(git ls-files '*.sh')
uv lock --check
.venv/bin/python -m pytest -q --cov=src --cov=api
uvx pip-audit -r requirements.txt
```

## 3. Сборка и локальный smoke

```bash
uv build
scripts/smoke_wheel.sh dist/decision_cascade-<версия>-py3-none-any.whl
```

Для релиза с весами — артефакты рядом (они не собираются в CI: веса лежат в
`checkpoints/`, которого нет в git):

```bash
NO_RELEASE=1 training/recipe/publish_weight_packs.sh v<версия> checkpoints/packs
eval/competitors/publish_benchmarks.sh v<версия>
```

## 4. Публикация

```bash
git tag -a v<версия> -m "decision-cascade <версия>: <кратко>"
git push origin main v<версия>
```

Тег запускает `.github/workflows/pypi.yml`: тесты → сборка → smoke wheel →
публикация на PyPI (если версии там ещё нет) → установка из индекса и проверка
`src.__version__`. Повторный запуск идемпотентен: опубликованную версию он не
перезаписывает.

Секрет `PYPI_API_TOKEN` (токен PyPI, scope — только проект `decision-cascade`)
задаётся в настройках репозитория; без него job не падает, а пишет
предупреждение и пропускает публикацию.

GitHub Release с артефактами собирается отдельно (веса и сырые прогоны
бенчмарков лежат локально):

```bash
gh release create v<версия> \
  dist/decision_cascade-<версия>-py3-none-any.whl \
  dist/decision_cascade-<версия>.tar.gz \
  dist/v<версия>-weights.tar.gz \
  dist/v<версия>-manifest.json \
  dist/v<версия>-benchmarks.tar.gz \
  --title "v<версия> — <кратко>" --notes-file notes.md
```

## 5. После релиза

- дождаться зелёного CI и job `Publish to PyPI`;
- проверить установку из индекса в чистом venv:
  ```bash
  pip install "decision-cascade==<версия>"
  python -c "import src; print(src.__version__)"
  ```
- ночной регресс берёт `*-weights.tar.gz` из последнего релиза, поэтому
  следующий прогон автоматически проверит новые паки;
- если релиз ломающий (major) — обновить `docs/stability.md` и объявить
  депрекации в реестре `api/server/deprecation.py`.

## 6. Если что-то пошло не так

- **Версия уже на PyPI.** Перезапись невозможна: выпускайте патч (`X.Y.Z+1`),
  тег с тем же номером не переиспользуйте.
- **CI красный после тега.** Тег можно удалить до публикации
  (`git push --delete origin v<версия>`), исправить и поставить заново; если
  пакет уже ушёл на PyPI — только новый патч.
- **Сломанный релиз в GitHub.** Отредактируйте notes, артефакты можно
  переприложить; сам тег не перезаписывайте.
