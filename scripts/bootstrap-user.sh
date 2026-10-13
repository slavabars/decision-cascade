#!/usr/bin/env bash
# User-space setup — NO sudo required
set -euo pipefail

echo "▶ Установка uv (user-space)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "▶ Установка pyenv (user-space)"
if ! command -v pyenv >/dev/null; then
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
  export PATH="$HOME/.pyenv/bin:$PATH"
fi

echo "▶ Создание venv"
uv venv .venv --python 3.11

echo "▶ Установка Python-зависимостей"
uv pip install --python .venv/bin/python -r requirements.txt

echo "▶ Установка pre-commit"
uv pip install --python .venv/bin/python pre-commit

echo "▶ Установка GitHub CLI (gh, user-space)"
if ! command -v gh >/dev/null; then
  gh_arch=""
  case "$(uname -m)" in
    x86_64|amd64) gh_arch="amd64" ;;
    aarch64|arm64) gh_arch="arm64" ;;
  esac
  if [ -z "$gh_arch" ]; then
    echo "   пропускаю: архитектура $(uname -m) не покрыта tarball'ом gh"
  else
    # Версия — из редиректа /releases/latest (без GitHub API и его лимитов).
    gh_version="$(curl -fsSLI -o /dev/null -w '%{url_effective}' \
      https://github.com/cli/cli/releases/latest 2>/dev/null \
      | sed -n 's#.*/v\([0-9][0-9.]*\)$#\1#p')"
    gh_version="${gh_version:-2.63.2}"
    gh_tmp="$(mktemp -d)"
    if curl -fsSL \
      "https://github.com/cli/cli/releases/download/v${gh_version}/gh_${gh_version}_linux_${gh_arch}.tar.gz" \
      | tar -xz -C "$gh_tmp"; then
      mkdir -p "$HOME/.local/bin"
      install -m 0755 \
        "$gh_tmp/gh_${gh_version}_linux_${gh_arch}/bin/gh" \
        "$HOME/.local/bin/gh"
      echo "   gh ${gh_version} -> $HOME/.local/bin/gh"
    else
      echo "   предупреждение: не удалось скачать gh ${gh_version}"
    fi
    rm -rf "$gh_tmp"
  fi
fi

echo "▶ Готово. Активируйте venv: source .venv/bin/activate"
