#!/usr/bin/env bash
# Публикует чекпоинт голов как артефакт GitHub Release (веса не коммитим в git).
# Usage: training/recipe/publish_release.sh <tag> <checkpoint-dir> [repo]
#
# Загружаются все *.pt из каталога (heads.pt + копия под подписью вопросов,
# которую сервис ищет в DECISION_CHECKPOINT_DIR) и metadata.json.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <tag> <checkpoint-dir> [repo]" >&2
  exit 2
fi

tag="$1"
dir="$2"
repo="${3:-}"

if [[ ! -f "$dir/heads.pt" ]]; then
  echo "missing $dir/heads.pt" >&2
  exit 1
fi

files=()
for checkpoint in "$dir"/*.pt; do
  [[ -e "$checkpoint" ]] && files+=("$checkpoint")
done
files+=("$dir/metadata.json")

args=(release create "$tag" "${files[@]}"
  --title "$tag" --notes "Decision heads $tag")
if [[ -n "$repo" ]]; then
  args+=(--repo "$repo")
fi

gh "${args[@]}"
echo "published $tag"
