#!/usr/bin/env bash
# Пакует дообученный энкодер вместе с головами и температурой в артефакт релиза.
#
# Usage: training/recipe/publish_encoder.sh <tag> <finetune-dir> [name]
#        NO_RELEASE=1 — только собрать dist/<tag>-encoder-<name>.tar.gz
#
# Каталог дообучения даёт `training/recipe/finetune_encoder.py`:
#   <dir>/encoder/  <dir>/heads.pt  <dir>/<signature>.pt  <dir>/metadata.json
#   <dir>/temperature.json (после calibrate_temperature)
#
# В архиве: encoder/, heads.pt, <signature>.pt, metadata.json, temperature.json,
# README.md с командой запуска и manifest.json (sha256, метрики, размеры).
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <tag> <finetune-dir> [name]" >&2
  exit 2
fi

tag="$1"
src="$2"
name="${3:-$(basename "$src")}"
python_bin="${PYTHON:-}"
if [[ -z "$python_bin" && -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi
python_bin="${python_bin:-python}"

[[ -d "$src/encoder" ]] || {
  echo "нет $src/encoder: это не каталог дообучения" >&2
  exit 1
}
[[ -f "$src/heads.pt" ]] || {
  echo "нет $src/heads.pt" >&2
  exit 1
}

signature="$("$python_bin" - "$src" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / "metadata.json"
print(json.loads(path.read_text(encoding="utf-8"))["questions_signature"])
PY
)"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
root="$stage/${tag}-encoder-${name}"
mkdir -p "$root"
cp -r "$src/encoder" "$root/encoder"
cp "$src/heads.pt" "$root/heads.pt"
if [[ -f "$src/$signature.pt" ]]; then
  cp "$src/$signature.pt" "$root/$signature.pt"
else
  cp "$src/heads.pt" "$root/$signature.pt"
fi
cp "$src/metadata.json" "$root/metadata.json"
if [[ -f "$src/temperature.json" ]]; then
  cp "$src/temperature.json" "$root/temperature.json"
fi

cat > "$root/README.md" <<EOF
# Дообученный энкодер \`${name}\`

Энкодер (${signature}) вместе с обученными головами и температурой. В отличие от
килобайтных паков (\`*-weights.tar.gz\`), здесь лежат веса самого энкодера —
поэтому архив большой.

## Запуск

\`\`\`bash
tar -xzf ${tag}-encoder-${name}.tar.gz
DECISION_ENCODER_KIND=encoder \\
DECISION_MODEL_ID=${tag}-encoder-${name}/encoder \\
DECISION_FEATURE_DIM=768 \\
DECISION_CHECKPOINT_DIR=${tag}-encoder-${name} \\
DECISION_MAX_LENGTH=256 \\
DECISION_DEVICE=cuda \\
  python -m uvicorn api.server.main:app --host 127.0.0.1 --port 8000
\`\`\`

\`DECISION_ENCODER_KIND=encoder\` обязателен: это encoder-only модель, а не causal
LM (в metadata пака вид энкодера не пишется). Головы сервис находит по подписи
вопросов \`${signature}.pt\`.

## Что внутри

- \`encoder/\` — веса энкодера в формате \`from_pretrained\` (fp32);
- \`heads.pt\`, \`${signature}.pt\` — головы решений (одно и то же);
- \`metadata.json\` — шаги, lr, сплиты, метрики train/val/test;
- \`temperature.json\` — калибровка по вопросам (argmax не меняется);
- \`manifest.json\` — размеры и sha256 файлов.
EOF

# manifest кладём в каталог ДО архивации, чтобы он был и внутри; хеш и размер
# самого архива добавляем в копию рядом с архивом после сборки
"$python_bin" - "$root" "$tag" "$name" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, tag, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
temperature = {}
if (root / "temperature.json").exists():
    temperature = json.loads(
        (root / "temperature.json").read_text(encoding="utf-8")
    ).get("temperature", {})
files = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
manifest = {
    "tag": tag,
    "name": name,
    "encoder": meta.get("encoder"),
    "encoder_kind": "encoder",
    "feature_dim": meta.get("feature_dim"),
    "specs": meta.get("specs"),
    "questions_signature": meta.get("questions_signature"),
    "steps": meta.get("steps"),
    "lr": meta.get("lr"),
    "seed": meta.get("seed"),
    "metrics": meta.get("metrics"),
    "temperature": temperature,
    "files": files,
}
(root / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

mkdir -p dist
archive="dist/${tag}-encoder-${name}.tar.gz"
tar -czf "$archive" -C "$stage" "${tag}-encoder-${name}"

"$python_bin" - "$root" "$archive" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, archive = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
manifest["archive"] = {
    "name": archive.name,
    "bytes": archive.stat().st_size,
    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
}
sidecar = archive.parent / f"{archive.name[:-len('.tar.gz')]}-manifest.json"
sidecar.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"собрано: {archive} ({archive.stat().st_size / 1e6:.0f} МБ)")
print(f"manifest: {sidecar}")
PY
