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

echo "▶ Готово. Активируйте venv: source .venv/bin/activate"
