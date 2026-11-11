#!/usr/bin/env bash
# Ставит Stockfish в .tools/ без sudo: официальный бинарник с GitHub.
#
#   bash examples/games/chess/fetch_engine.sh
#
# Нужен curl и tar. Версия — последний релиз; для воспроизводимости номера
# партий это не важно, но если нужна конкретная — задайте TAG=sf_17.
set -euo pipefail

cd "$(dirname "$0")/../../.."

DEST="${DEST:-.tools}"
TAG="${TAG:-latest}"
ARCH="${ARCH:-linux-x86-64}"
API="https://api.github.com/repos/official-stockfish/Stockfish/releases/${TAG}"

command -v curl >/dev/null || {
  echo "нужен curl" >&2
  exit 1
}

mkdir -p "$DEST"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

url="$(curl -fsSL "$API" | python3 -c '
import json, sys
release = json.load(sys.stdin)
for asset in release["assets"]:
    if sys.argv[1] in asset["name"]:
        print(asset["browser_download_url"])
        break
else:
    sys.exit("нет ассета с " + sys.argv[1])
' "$ARCH")"

echo "качаю: $url"
curl -fsSL -o "$work/stockfish.tar.gz" "$url"
tar -xzf "$work/stockfish.tar.gz" -C "$work"

binary="$(find "$work" -maxdepth 3 -type f -name "*${ARCH}*" ! -name "*.tar.gz" | head -1)"
[[ -n "$binary" ]] || {
  echo "в архиве не найден бинарник" >&2
  exit 1
}

install -m 0755 "$binary" "$DEST/stockfish"
printf 'uci\nquit\n' | "$DEST/stockfish" 2>/dev/null | grep -m1 "id name" || true
echo "готово: $DEST/stockfish"
