"""Публикация паков весов на Hugging Face Hub.

Пак — это обученные головы решений (``heads.pt``), ``metadata.json`` с
подписью вопросов, сплитами и метриками, и ``temperature.json`` с отчётом
калибровки. Веса энкодера в пак не входят: он задаётся ``DECISION_MODEL_ID`` и
живёт отдельной моделью на Хабе. Поэтому пак — не ``transformers``-модель, и
карточка это проговаривает.

Команды:

    # проверить паки и собрать манифест, ничего не загружая
    python -m training.recipe.publish_hf --check

    # собрать staging-каталог: manifest.json, README.md, packs/<имя>/...
    python -m training.recipe.publish_hf --dry-run --out dist/hf

    # опубликовать и поставить тег-ревизию
    HF_TOKEN=... python -m training.recipe.publish_hf \
        --publish --repo slavabars/decision-cascade-packs --revision v1.8.0

    # сверить опубликованное с манифестом
    python -m training.recipe.publish_hf --verify \
        --repo slavabars/decision-cascade-packs --revision v1.8.0

Токен читается только из ``HF_TOKEN`` и никогда не печатается. Повторная
публикация той же ревизии не перезаписывает её: если тег уже есть, файлы
сверяются по хэшам, и расхождение — ошибка («поднимите ревизию»).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src import __version__

SCHEMA_VERSION = "1"
LIBRARY = "decision-cascade"
DEFAULT_REPO = "slavabars/decision-cascade-packs"
DEFAULT_PACKS = Path("checkpoints/packs")
DEFAULT_OUT = Path("dist/hf")
#: Файлы, которые нужны сервису или служат отчётом о калибровке. Датасеты
#: (``dataset.jsonl``) не публикуем: они воспроизводятся из публичных наборов.
PACK_FILES = ("heads.pt", "metadata.json", "temperature.json")
#: Паки с неясными условиями датасета не публикуем — как и в GitHub-релизе.
DEFAULT_EXCLUDES = ("chess", "maze", "ru-rubert-tiny2")
#: Происхождение датасета для карточки: ссылка на HF и условия.
DATASET_SOURCES: dict[str, dict[str, str | None]] = {
    "ag_news": {"hub": "fancyzhx/ag_news", "license": "unknown"},
    "routerbench_0shot": {"hub": "withmartian/routerbench", "license": "not declared"},
    "support": {"hub": None, "license": "synthetic, Apache-2.0 (this project)"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_entry(pack_dir: Path) -> dict[str, Any]:
    """Один пак -> запись манифеста с хэшами файлов и тестовыми метриками."""
    metadata_path = pack_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    encoder = metadata.get("encoder")
    feature_dim = metadata.get("feature_dim")
    signature = metadata.get("questions_signature")
    if not encoder:
        raise ValueError(f"{pack_dir.name}: metadata.json has no encoder")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError(f"{pack_dir.name}: bad feature_dim {feature_dim!r}")
    if not signature:
        raise ValueError(f"{pack_dir.name}: metadata.json has no questions_signature")
    heads = pack_dir / "heads.pt"
    if not heads.is_file():
        raise ValueError(f"{pack_dir.name}: heads.pt is missing")

    files: dict[str, dict[str, Any]] = {}
    for name in PACK_FILES:
        path = pack_dir / name
        if path.is_file():
            files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    questions = metadata.get("questions") or {}
    dataset_name = metadata.get("dataset_name")
    source = DATASET_SOURCES.get(dataset_name, {"hub": None, "license": "see upstream"})
    return {
        "name": pack_dir.name,
        "encoder": encoder,
        "feature_dim": feature_dim,
        "questions_signature": signature,
        "questions": {
            name: (value.get("type") if isinstance(value, dict) else str(value))
            for name, value in questions.items()
        },
        "dataset": dataset_name,
        "dataset_source": source["hub"],
        "dataset_license": source["license"],
        "splits": metadata.get("splits") or {},
        "metrics": {"test": (metadata.get("metrics") or {}).get("test") or {}},
        "files": files,
    }


def build_manifest(
    packs_root: Path,
    *,
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
    version: str = __version__,
    commit: str | None = None,
) -> dict[str, Any]:
    """Собрать манифест из каталога паков; исключения — по имени каталога."""
    if not packs_root.is_dir():
        raise ValueError(f"packs root does not exist: {packs_root}")
    packs = []
    for metadata in sorted(packs_root.glob("*/metadata.json")):
        name = metadata.parent.name
        if name in excludes:
            continue
        packs.append(pack_entry(metadata.parent))
    if not packs:
        raise ValueError(f"no packs to publish in {packs_root}")
    return {
        "schema_version": SCHEMA_VERSION,
        "library": LIBRARY,
        "library_version": version,
        "git_commit": commit,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "packs": packs,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pack_metric(pack: dict[str, Any], name: str) -> float | None:
    questions = (pack.get("metrics", {}).get("test") or {}).get("questions") or {}
    values = [
        float(report[name])
        for report in questions.values()
        if isinstance(report, dict) and report.get(name) is not None
    ]
    return _mean(values)


def _accuracy(pack: dict[str, Any]) -> float | None:
    test = pack.get("metrics", {}).get("test") or {}
    if test.get("overall_accuracy") is not None:
        return float(test["overall_accuracy"])
    return _pack_metric(pack, "accuracy")


def render_card(
    manifest: dict[str, Any],
    *,
    repo: str = DEFAULT_REPO,
    revision: str | None = None,
) -> str:
    """Карточка репозитория: метаданные, загрузка, таблица паков, оговорки."""
    datasets = sorted(
        {pack["dataset_source"] for pack in manifest["packs"] if pack["dataset_source"]}
    )
    lines = [
        "---",
        "license: apache-2.0",
        "language:",
        "- en",
        "- ru",
        "tags:",
        "- decision-cascade",
        "- calibrated-decisions",
        "- selective-prediction",
        "- risk-aware",
        "- decision-making",
        "- model-routing",
        "- llm-routing",
        "- model-cascade",
        "- local-first",
        "- cost-efficient-inference",
        "- uncertainty-estimation",
        "- conformal-prediction",
        "- systemone",
    ]
    if datasets:
        lines.append("datasets:")
        lines.extend(f"- {name}" for name in datasets)
    lines += [
        "---",
        "",
        "# decision-cascade weight packs",
        "",
        (
            "Trained decision heads for "
            "[decision-cascade](https://github.com/slavabars/decision-cascade): "
            "typed decisions (Noul / Choice / Score) with calibrated confidence, "
            "escalation thresholds and an optional risk model."
        ),
        "",
        (
            "> **This is not a `transformers` model.** There is no "
            "`AutoModel.from_pretrained` or `pipeline(\"decision\")` here. "
            "The repository ships head weights plus a manifest; the base encoder "
            "is a separate model referenced by `DECISION_MODEL_ID`."
        ),
        "",
        f"Library version: `{manifest['library_version']}`"
        + (
            f", commit `{manifest['git_commit']}`"
            if manifest.get("git_commit")
            else ""
        ),
        "",
        "## Install",
        "",
        "```bash",
        "pip install decision-cascade",
        "pip install huggingface_hub",
        "```",
        "",
        "## Download and run",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "",
        "path = snapshot_download(",
        f'    "{repo}",',
    ]
    if revision:
        lines.append(f'    revision="{revision}",')
    lines += [
        '    allow_patterns=["packs/public-gpt2/*"],',
        ")",
        "print(path)",
        "```",
        "",
        "```bash",
        "DECISION_MODEL_ID=openai-community/gpt2 \\",
        "DECISION_CHECKPOINT_DIR=<path>/packs/public-gpt2 \\",
        "  uvicorn api.server.main:app --port 8000",
        "```",
        "",
        (
            "Pointing `DECISION_CHECKPOINT_DIR` at the whole `packs/` directory "
            "serves every pack at once; the service picks one by the question "
            "signature and feature dimension."
        ),
        "",
        "## Packs",
        "",
        "| Pack | Encoder | Dim | Dataset | Test accuracy | Mean Brier | Mean ECE |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for pack in manifest["packs"]:
        accuracy = _accuracy(pack)
        brier = _pack_metric(pack, "brier")
        ece = _pack_metric(pack, "ece")
        lines.append(
            "| `{name}` | `{encoder}` | {dim} | {dataset} | {acc} | {brier} | {ece} |".format(
                name=pack["name"],
                encoder=pack["encoder"],
                dim=pack["feature_dim"],
                dataset=pack.get("dataset") or "—",
                acc="—" if accuracy is None else f"{accuracy:.3f}",
                brier="—" if brier is None else f"{brier:.3f}",
                ece="—" if ece is None else f"{ece:.3f}",
            )
        )
    lines += [
        "",
        (
            "Metrics are measured on each pack's test split and stored in "
            "`manifest.json`; they say nothing about your domain. Calibrate on "
            "your own labelled data before trusting escalation rates."
        ),
        "",
        "## Questions per pack",
        "",
    ]
    for pack in manifest["packs"]:
        questions = ", ".join(
            f"`{name}` ({kind})" for name, kind in pack["questions"].items()
        )
        lines.append(f"- `{pack['name']}`: {questions or '—'}")
    lines += [
        "",
        "## Data and license",
        "",
        (
            "The code and the head weights are Apache-2.0. Dataset terms are "
            "separate:"
        ),
        "",
    ]
    for pack in manifest["packs"]:
        source = pack.get("dataset_source") or (
            "synthetic" if pack.get("dataset") == "support" else "upstream dataset"
        )
        lines.append(
            f"- `{pack['name']}`: {pack.get('dataset') or '—'} — "
            f"{source}, license: {pack.get('dataset_license') or 'see upstream'}."
        )
    lines += [
        "",
        (
            "Benchmark records are used to train and evaluate the heads; the "
            "dataset files themselves are not redistributed here."
        ),
        "",
        "## Reproducibility",
        "",
        (
            "`manifest.json` pins the library version, the git commit and the "
            "SHA-256 of every file. The verification command:"
        ),
        "",
        "```bash",
        "python -m training.recipe.publish_hf --verify \\",
        f'    --repo {repo}' + (f" --revision {revision}" if revision else ""),
        "```",
        "",
    ]
    return "\n".join(lines)


def stage(
    packs_root: Path,
    out_dir: Path,
    manifest: dict[str, Any],
    *,
    repo: str = DEFAULT_REPO,
    revision: str | None = None,
) -> Path:
    """Собрать staging-каталог: manifest.json, README.md и packs/<имя>/."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for pack in manifest["packs"]:
        target = out_dir / "packs" / pack["name"]
        target.mkdir(parents=True)
        for name in pack["files"]:
            shutil.copy2(packs_root / pack["name"] / name, target / name)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        render_card(manifest, repo=repo, revision=revision), encoding="utf-8"
    )
    return out_dir


def check_manifest(manifest: dict[str, Any]) -> list[str]:
    """Предупреждения, которые не мешают публикации, но видны в отчёте."""
    warnings = []
    seen: dict[tuple[str, int], list[str]] = {}
    datasets: dict[str, str | None] = {}
    for pack in manifest["packs"]:
        key = (pack["questions_signature"], int(pack["feature_dim"]))
        seen.setdefault(key, []).append(pack["name"])
        datasets.setdefault(pack.get("dataset") or "—", pack.get("dataset_license"))
    for dataset, license_name in sorted(datasets.items()):
        if license_name in (None, "see upstream", "unknown", "not declared"):
            warnings.append(
                f"dataset {dataset}: license is {license_name or 'not described'}; "
                "the card will say so, check the terms before publishing"
            )
    for (signature, dim), names in seen.items():
        if len(names) > 1:
            warnings.append(
                f"signature {signature} (dim {dim}) is shared by "
                f"{', '.join(names)}; in one checkpoint dir the service will "
                "pick one by sort order"
            )
    return warnings


def resolve_token() -> str | None:
    """Токен из ``HF_TOKEN`` или из локального логина ``huggingface-cli login``.

    Секрет не печатается и не попадает в манифест; при отсутствии обоих
    источников публикация завершается ошибкой до сетевых вызовов.
    """
    value = os.environ.get("HF_TOKEN", "").strip()
    if value:
        return value
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def _tag_exists(api: Any, repo: str, revision: str) -> bool:
    refs = api.list_repo_refs(repo_id=repo)
    return any(tag.name == revision for tag in refs.tags)


def verify_published(
    repo: str,
    revision: str | None,
    *,
    expected: dict[str, Any] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Скачать опубликованный манифест и сверить хэши всех файлов."""
    import tempfile

    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=tmp,
            token=token or None,
            allow_patterns=["manifest.json", "packs/*/*"],
        )
        remote = json.loads(
            (Path(tmp) / "manifest.json").read_text(encoding="utf-8")
        )
        if expected is not None:
            if remote["schema_version"] != expected["schema_version"]:
                raise RuntimeError("published manifest has a different schema version")
            if remote["packs"] != expected["packs"]:
                raise RuntimeError(
                    "published packs differ from the local build; "
                    "bump --revision instead of overwriting a tag"
                )
        for pack in remote["packs"]:
            for name, info in pack["files"].items():
                path = Path(tmp) / "packs" / pack["name"] / name
                if not path.is_file():
                    raise RuntimeError(f"published pack is missing {path}")
                if sha256(path) != info["sha256"]:
                    raise RuntimeError(f"sha256 mismatch for {path}")
    return remote


def _verify_with_retries(
    repo: str,
    revision: str | None,
    *,
    expected: dict[str, Any],
    token: str | None,
) -> None:
    """Сверка после загрузки: тег и файлы на Хабе появляются не мгновенно."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            verify_published(
                repo, revision, expected=expected, token=token
            )
            return
        except Exception as exc:  # noqa: BLE001 - повторяем любую сетевую ошибку
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    assert last_error is not None
    raise last_error


def publish(
    staging: Path,
    manifest: dict[str, Any],
    *,
    repo: str,
    revision: str | None,
    token: str,
) -> None:
    """Загрузить staging на Hub; тег-ревизия создаётся один раз."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    if revision and _tag_exists(api, repo, revision):
        _verify_with_retries(repo, revision, expected=manifest, token=token)
        print(f"{repo}@{revision} already published and matches the local build")
        return
    api.upload_folder(
        folder_path=staging,
        repo_id=repo,
        repo_type="model",
        commit_message=f"packs: {manifest['library_version']}",
    )
    if revision:
        api.create_tag(repo_id=repo, tag=revision, revision="main")
    _verify_with_retries(repo, revision, expected=manifest, token=token)
    print(f"published {repo}" + (f"@{revision}" if revision else ""))


def update_card(
    manifest: dict[str, Any],
    *,
    repo: str,
    revision: str | None,
    token: str,
    target_revision: str = "main",
) -> None:
    """Обновить только ``README.md`` на ``main``, не трогая паки и теги."""
    from huggingface_hub import HfApi

    card = render_card(manifest, repo=repo, revision=revision)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        revision=target_revision,
        commit_message="card: discovery tags for routing, local-first and SystemOne",
    )
    print(f"card updated: {repo}@{target_revision}")


def _load_stage(stage: Path) -> dict[str, Any]:
    """Прочитать staging и проверить, что файлы совпадают с манифестом."""
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    for pack in manifest["packs"]:
        for name, info in pack["files"].items():
            path = stage / "packs" / pack["name"] / name
            if not path.is_file():
                raise RuntimeError(f"staging is missing {path}")
            if sha256(path) != info["sha256"]:
                raise RuntimeError(f"sha256 mismatch in staging for {path}")
    return manifest


def _from_stage(args: argparse.Namespace) -> int:
    """Публикация или проверка уже собранного staging-каталога."""
    try:
        manifest = _load_stage(args.from_stage)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"stage {args.from_stage}: {len(manifest['packs'])} packs, "
        f"library {manifest['library_version']}"
    )
    if args.verify:
        verify_published(args.repo, args.revision, expected=manifest)
        print("staging matches the published files")
        return 0
    token = resolve_token()
    if not token:
        print(
            "error: not authenticated: set HF_TOKEN or run `huggingface-cli login`",
            file=sys.stderr,
        )
        return 2
    publish(
        args.from_stage,
        manifest,
        repo=args.repo,
        revision=args.revision,
        token=token,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish decision-cascade packs")
    parser.add_argument("--packs", type=Path, default=DEFAULT_PACKS)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=None, help="тег-ревизия, например v1.8.0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_EXCLUDES),
        help="паки, которые не публикуются (по имени каталога)",
    )
    parser.add_argument("--check", action="store_true", help="только проверить")
    parser.add_argument(
        "--dry-run", action="store_true", help="собрать staging без загрузки"
    )
    parser.add_argument("--publish", action="store_true", help="загрузить на Hub")
    parser.add_argument(
        "--update-card",
        action="store_true",
        help="обновить только README.md на main",
    )
    parser.add_argument("--verify", action="store_true", help="сверить с Hub")
    parser.add_argument(
        "--from-stage",
        type=Path,
        default=None,
        help="опубликовать/сверить уже собранный staging-каталог",
    )
    parser.add_argument(
        "--library-version",
        default=__version__,
        help="версия библиотеки/релиза, к которой относятся паки",
    )
    parser.add_argument(
        "--commit", default=None, help="git commit, на котором собраны паки"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    actions = [
        args.check,
        args.dry_run,
        args.publish,
        args.verify,
        args.update_card,
    ]
    if sum(bool(action) for action in actions) > 1:
        raise SystemExit("choose one of --check, --dry-run, --publish, --verify")
    if not any(actions):
        args.check = True
    if args.from_stage is not None and not (args.publish or args.verify):
        raise SystemExit("--from-stage requires --publish or --verify")
    if args.verify:
        remote = verify_published(args.repo, args.revision)
        print(
            f"{args.repo}"
            + (f"@{args.revision}" if args.revision else "")
            + f": {len(remote['packs'])} packs, hashes OK"
        )
        return 0
    if args.from_stage is not None:
        return _from_stage(args)

    try:
        manifest = build_manifest(
            args.packs,
            excludes=tuple(args.exclude),
            version=args.library_version,
            commit=args.commit,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    warnings = check_manifest(manifest)
    total = sum(
        info["bytes"]
        for pack in manifest["packs"]
        for info in pack["files"].values()
    )
    print(
        f"{len(manifest['packs'])} packs, {total} bytes, "
        f"library {manifest['library_version']}"
    )
    for pack in manifest["packs"]:
        print(f"  {pack['name']:<24} {pack['encoder']:<24} dim={pack['feature_dim']}")
    for warning in warnings:
        print(f"warning: {warning}")

    if args.check:
        return 0
    if args.update_card:
        token = resolve_token()
        if not token:
            print(
                "error: not authenticated: set HF_TOKEN or run `huggingface-cli login`",
                file=sys.stderr,
            )
            return 2
        update_card(
            manifest,
            repo=args.repo,
            revision=args.revision,
            token=token,
        )
        return 0
    staging = stage(
        args.packs,
        args.out,
        manifest,
        repo=args.repo,
        revision=args.revision,
    )
    print(f"staging -> {staging}")
    if args.dry_run:
        return 0
    token = resolve_token()
    if not token:
        print(
            "error: not authenticated: set HF_TOKEN or run `huggingface-cli login`",
            file=sys.stderr,
        )
        return 2
    publish(
        staging,
        manifest,
        repo=args.repo,
        revision=args.revision,
        token=token,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
