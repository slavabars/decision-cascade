#!/usr/bin/env bash
# Публикует набор обученных паков весов одним артефактом GitHub Release.
# Usage: training/recipe/publish_weight_packs.sh <tag> <packs-root> [repo]
#        NO_RELEASE=1 — только собрать артефакты в dist/, не создавать релиз.
#
# Собирает каталог паков в архив <tag>-weights.tar.gz (без датасетов *.jsonl,
# они не нужны сервису) плюс manifest.json со сводкой метрик и подписей, и
# создаёт GitHub Release. Репозиторий берётся из origin, если не задан явно.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <tag> <packs-root> [repo]" >&2
  exit 2
fi

tag="$1"
root="$2"
repo="${3:-}"
python_bin="${PYTHON:-}"
if [[ -z "$python_bin" && -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi
python_bin="${python_bin:-python}"

[[ -d "$root" ]] || {
  echo "no such packs root: $root" >&2
  exit 1
}

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

cp -r "$root" "$stage/${tag}-weights"
# Веса сервису нужны без датасетов — не раздуваем архив.
find "$stage/${tag}-weights" -name '*.jsonl' -delete

"$python_bin" - "$stage/${tag}-weights" <<'PY' > "$stage/manifest.json"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
packs = []
for meta_path in sorted(root.glob("*/metadata.json")):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    packs.append({
        "pack": meta_path.parent.name,
        "encoder": meta.get("encoder"),
        "feature_dim": meta.get("feature_dim"),
        "specs": meta.get("specs"),
        "dataset_name": meta.get("dataset_name"),
        "questions_signature": meta.get("questions_signature"),
        "temperature": meta.get("temperature"),
        "metrics": {
            name: report.get("overall_accuracy")
            for name, report in meta.get("metrics", {}).items()
        },
    })
print(json.dumps({"schema_version": "1", "packs": packs},
                 ensure_ascii=False, indent=2))
PY

cp "$stage/manifest.json" "$stage/${tag}-weights/manifest.json"

mkdir -p dist
tar -czf "dist/${tag}-weights.tar.gz" -C "$stage" "${tag}-weights"
cp "$stage/manifest.json" "dist/${tag}-manifest.json"

if [[ "${NO_RELEASE:-0}" == "1" ]]; then
  echo "собрано без публикации:"
  echo "  dist/${tag}-weights.tar.gz"
  echo "  dist/${tag}-manifest.json"
  exit 0
fi

args=(release create "$tag"
  "dist/${tag}-weights.tar.gz"
  "dist/${tag}-manifest.json"
  --title "$tag weights"
  --notes "Trained decision-head packs ($tag). Распакуйте архив и укажите DECISION_CHECKPOINT_DIR на нужный пак.")
if [[ -n "$repo" ]]; then
  args+=(--repo "$repo")
fi

gh "${args[@]}"
echo "published $tag"
