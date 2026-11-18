# Release: version, PyPI and artifacts

**English** · [Русский](../release.md)

The process is the same for a patch, a minor and a major version; only the size of the CHANGELOG and the decision "do we break the contract" differ (breaking is allowed only in a major).

## 1. Preparation

1. **Version** — `src/__init__.py` (the single source: `pyproject.toml` takes it through `dynamic = ["version"]`).
2. **CHANGELOG** — a section `## [X.Y.Z] — YYYY-MM-DD` and a link `[X.Y.Z]: .../compare/v<previous>...vX.Y.Z`; `[Unreleased]` switches to comparing against the new version.
3. **Documentation** — if behaviour changes: README, `docs/*.md`, `docs/stability.md` (the surface status: stable or beta). Links and images in `README.md` must be absolute: PyPI renders the page from the uploaded description, and relative paths there lead to 404.
4. **Tests** — `tests/test_release.py` checks that the version, README, CHANGELOG and package classifier agree. A mismatch is a red CI.

## 2. Checks before tagging

```bash
.venv/bin/python -m ruff check src api eval training tests examples scripts
shellcheck -x $(git ls-files '*.sh')
uv lock --check
.venv/bin/python -m pytest -q --cov=src --cov=api
uvx pip-audit -r requirements.txt
```

## 3. Build and local smoke

```bash
uv build
scripts/smoke_wheel.sh dist/decision_cascade-<version>-py3-none-any.whl
```

For a release with weights, build the side artifacts (they are not built in CI: the weights live in `checkpoints/`, which is not in git):

```bash
NO_RELEASE=1 training/recipe/publish_weight_packs.sh v<version> checkpoints/packs
eval/competitors/publish_benchmarks.sh v<version>
```

`EXCLUDE_PACKS` (comma-separated) removes packs from the archive — for example, if the dataset does not allow publishing derivative weights:

```bash
EXCLUDE_PACKS=ru-rubert-tiny2 NO_RELEASE=1 \
  training/recipe/publish_weight_packs.sh v<version> checkpoints/packs
```

## 4. Publication

```bash
git tag -a v<version> -m "decision-cascade <version>: <summary>"
git push origin main v<version>
```

The tag triggers `.github/workflows/pypi.yml`: tests → build → smoke wheel → publish to PyPI (if the version is not there yet) → install from the index and check `src.__version__`. A repeated run is idempotent: it does not overwrite a published version.

The `PYPI_API_TOKEN` secret (a PyPI token scoped only to the `decision-cascade` project) is configured in the repository settings; without it the job does not fail but writes a warning and skips publication.

The GitHub Release with artifacts is created separately (weights and raw benchmark runs live locally):

```bash
gh release create v<version> \
  dist/decision_cascade-<version>-py3-none-any.whl \
  dist/decision_cascade-<version>.tar.gz \
  dist/v<version>-weights.tar.gz \
  dist/v<version>-manifest.json \
  dist/v<version>-benchmarks.tar.gz \
  --title "v<version> — <summary>" --notes-file notes.md
```

## 5. After the release

- wait for green CI and the `Publish to PyPI` job;
- open the new PyPI page and check that the description is updated, images load and links point to GitHub;
- verify installation from the index in a clean venv:
  ```bash
  pip install "decision-cascade==<version>"
  python -c "import src; print(src.__version__)"
  ```
- the nightly regression takes `*-weights.tar.gz` from the latest release, so the next run automatically checks the new packs;
- if the release is breaking (major) — update `docs/stability.md` and announce deprecations in the `api/server/deprecation.py` registry.

## 6. If something goes wrong

- **The version is already on PyPI.** Overwriting is impossible: ship a patch (`X.Y.Z+1`), and do not reuse a tag with the same number.
- **CI is red after the tag.** The tag can be deleted before publication (`git push --delete origin v<version>`), fixed and created again; if the package has already reached PyPI — only a new patch.
- **A broken GitHub release.** Edit the notes and re-attach artifacts; do not overwrite the tag itself.

## 7. How to write release text, commits and documentation

A release is read by a person in a browser, not in a terminal, so:

- **A paragraph is one line.** No hard wraps at 80 characters — neither in the release description, nor in commits, nor in README and `docs/`: GitHub turns a single line break into `<br>`, and the paragraph falls apart into ragged pieces. The same text in a file stays readable because editors wrap lines themselves. The exception is code and docstrings: there line breaks are needed.
- **Write like a human for a human.** Get to the point: what changed, why, which numbers, what to do on upgrade. No meta phrases ("this release is about…"), no retelling of your own actions and no bureaucratic language.
- **Numbers only from real runs** (and say where to re-check them).
- **Structure** — short `##` sections: change, compatibility, checks, artifacts. Tables and lists are fine, but a paragraph of text is always one line.

Check before publication:

```bash
gh release view v<version> --json body -q .body | awk 'length($0) > 0 && length($0) < 60'
```
