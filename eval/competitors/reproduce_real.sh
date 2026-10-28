#!/usr/bin/env bash
# Воспроизводит сравнение с реальными релизами Clef-Flash и Laya.
#
# Поднимает участников, обучает и калибрует паки голов (если их нет), гоняет
# `eval.competitors.run` на трёх наборах и складывает JSON в $OUT. Все команды
# и параметры — те же, что зафиксированы в docs/competitor-benchmark.md.
#
# Usage:
#   eval/competitors/reproduce_real.sh [--specs support,public,routerbench] [--limit 200]
#
# Окружение (всё необязательное):
#   OUT=data/competitors_real        куда писать JSON-отчёты
#   PACKS=checkpoints/packs          каталог паков голов
#   SUPPORT_CKPT=checkpoints/reference
#   CLEF_GGUF=/path/clef-q4km.gguf   GGUF-релиз Clef-Flash для llama-server
#   LLAMA_BIN=/path/llama-server     бинарь llama.cpp (b11430+)
#   LLAMA_LD_PATH=/path/cudart       каталог с libcudart.so (если нужен)
#   LAYA_PYTHONPATH=/path/laya_pkg   распакованный laya (uv pip install --target)
#   ENCODER=openai-community/gpt2, DIM=768, DEVICE=cuda, MAX_LENGTH=256
#   SKIP_TRAIN=1                     не обучать/не калибровать паки
#   SKIP_CLEF=1 | SKIP_LAYA=1        не поднимать участника
#
# Пример:
#   CLEF_GGUF=/tmp/opencode/competitors/llama/clef-q4km.gguf \
#   LLAMA_BIN=/tmp/opencode/competitors/llama/bin/llama-b11430/llama-server \
#   LLAMA_LD_PATH=/tmp/opencode/competitors/llama/bin/llama-b11430/../cudart-llama-b11430-bin-ubuntu-cuda-13.4-x64 \
#   LAYA_PYTHONPATH=/tmp/opencode/competitors/laya_pkg \
#     eval/competitors/reproduce_real.sh
set -euo pipefail

specs="support,public,routerbench"
n=200
while [[ $# -gt 0 ]]; do
  case "$1" in
    --specs) specs="$2"; shift 2 ;;
    --n) n="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" && -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi
python_bin="${python_bin:-python}"

out="${OUT:-data/competitors_real}"
packs="${PACKS:-checkpoints/packs}"
support_ckpt="${SUPPORT_CKPT:-checkpoints/reference}"
encoder="${ENCODER:-openai-community/gpt2}"
dim="${DIM:-768}"
device="${DEVICE:-cuda}"
max_length="${MAX_LENGTH:-256}"
clef_port="${CLEF_PORT:-8091}"
laya_port="${LAYA_PORT:-8092}"
llama_port="${LLAMA_PORT:-8093}"
log_dir="${LOG_DIR:-/tmp/opencode/competitors}"
mkdir -p "$out" "$log_dir"

export DECISION_MAX_LENGTH="$max_length"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

wait_health() {
  local url="$1" name="$2"
  for _ in $(seq 1 120); do
    if curl -sf --max-time 2 "$url/health" >/dev/null; then
      echo "  $name готов: $url"
      return 0
    fi
    sleep 1
  done
  echo "  $name не поднялся: $url (см. $log_dir)" >&2
  return 1
}

# --- 1. паки голов -----------------------------------------------------------
pack_for() {
  case "$1" in
    support) echo "$support_ckpt" ;;
    public) echo "$packs/public-gpt2" ;;
    routerbench) echo "$packs/routerbench-gpt2" ;;
    *) echo "неизвестный набор: $1" >&2; exit 2 ;;
  esac
}

# `--limit` в eval.competitors.run — это размер датасета, а test = limit/4 у
# public и routerbench и limit у support. Здесь `--n` — целевой размер test.
limit_for() {
  case "$1" in
    support) echo "$n" ;;
    *) echo $((n * 4)) ;;
  esac
}

train_pack() {
  local spec="$1" dir="$2" extra=()
  [[ -f "$dir/heads.pt" ]] && { echo "  пак есть: $dir"; return 0; }
  echo "▶ обучение пака $dir"
  case "$spec" in
    support) extra=(--generate 1000 --train-steps 500 --rlcd-steps 300) ;;
    public) extra=(--limit 4000 --train-steps 500) ;;
    routerbench) extra=(--limit 4000 --train-steps 500 --rlcd-steps 200) ;;
  esac
  DECISION_DEVICE="$device" "$python_bin" -m training.recipe.train_heads \
    --specs "$spec" --out-dir "$dir" --encoder "$encoder" --dim "$dim" \
    --seed 0 "${extra[@]}"
}

calibrate_pack() {
  local spec="$1" dir="$2"
  [[ -f "$dir/temperature.json" ]] && { echo "  калибровка есть: $dir"; return 0; }
  echo "▶ калибровка $dir"
  local lim=4000
  [[ "$spec" == "support" ]] && lim=1000
  DECISION_DEVICE="$device" "$python_bin" -m training.recipe.calibrate_temperature \
    --specs "$spec" --encoder "$encoder" --dim "$dim" --limit "$lim" \
    --checkpoint-dir "$dir"
}

IFS=',' read -r -a spec_list <<< "$specs"
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "== паки голов"
  for spec in "${spec_list[@]}"; do
    dir="$(pack_for "$spec")"
    train_pack "$spec" "$dir"
    calibrate_pack "$spec" "$dir"
  done
fi

# --- 2. участники ------------------------------------------------------------
echo "== участники"
remotes=()
if [[ "${SKIP_CLEF:-0}" != "1" ]]; then
  if [[ -n "${CLEF_GGUF:-}" ]]; then
    [[ -n "${LLAMA_BIN:-}" ]] || { echo "нужен LLAMA_BIN" >&2; exit 2; }
    LD_LIBRARY_PATH="$(dirname "$LLAMA_BIN"):${LLAMA_LD_PATH:-}" \
      "$LLAMA_BIN" -m "$CLEF_GGUF" --host 127.0.0.1 --port "$llama_port" \
        -ngl 99 -c 16384 -ub 2048 --no-warmup --no-webui -t 8 \
        > "$log_dir/llama-server.log" 2>&1 &
    pids+=("$!")
    wait_health "http://127.0.0.1:$llama_port" "llama-server"
  fi
  clef_args=(--model clef-flash --port "$clef_port")
  if [[ -n "${CLEF_GGUF:-}" ]]; then
    clef_args+=(--base-url "http://127.0.0.1:$llama_port")
  fi
  "$python_bin" -m eval.competitors.release "${clef_args[@]}" \
    > "$log_dir/clef_proxy.log" 2>&1 &
  pids+=("$!")
  wait_health "http://127.0.0.1:$clef_port" "clef-flash"
  remotes+=("clef-flash=http://127.0.0.1:$clef_port")
fi
if [[ "${SKIP_LAYA:-0}" != "1" ]]; then
  [[ -n "${LAYA_PYTHONPATH:-}" ]] || { echo "нужен LAYA_PYTHONPATH" >&2; exit 2; }
  PYTHONPATH="$LAYA_PYTHONPATH" "$python_bin" -m eval.competitors.release \
    --model laya --device "$device" --port "$laya_port" \
    > "$log_dir/laya_server.log" 2>&1 &
  pids+=("$!")
  wait_health "http://127.0.0.1:$laya_port" "laya"
  remotes+=("laya=http://127.0.0.1:$laya_port")
fi
[[ ${#remotes[@]} -gt 0 ]] || { echo "нет ни одного участника (SKIP_CLEF/SKIP_LAYA)" >&2; exit 2; }

# --- 3. замеры ---------------------------------------------------------------
echo "== замеры (n=$n)"
for spec in "${spec_list[@]}"; do
  ckpt="$(pack_for "$spec")"
  lim="$(limit_for "$spec")"
  # По одному участнику за прогон: eval.competitors.run строит каскад только к
  # первому --remote, поэтому «все сразу» дало бы каскад лишь к одному из них.
  for r in "${remotes[@]}"; do
    name="${r%%=*}"
    echo "▶ $spec vs $name"
    "$python_bin" -m eval.competitors.run --specs "$spec" --limit "$lim" \
      --remote "$r" --encoder "$encoder" --dim "$dim" \
      --checkpoint-dir "$ckpt" --device "$device" \
      --out "$out/${spec}_pack_${name}.json"
  done
done

# Кривая «качество против доли эскалаций» — порог эскалации подбирается на домене.
for spec in "${spec_list[@]}"; do
  name="${remotes[0]%%=*}"
  echo "▶ sweep $spec vs $name"
  "$python_bin" -m eval.competitors.run --specs "$spec" \
    --limit "$(limit_for "$spec")" \
    --remote "${remotes[0]}" --encoder "$encoder" --dim "$dim" \
    --checkpoint-dir "$(pack_for "$spec")" --device "$device" --sweep \
    --out "$out/${spec}_pack_${name}_sweep.json"
done

echo
echo "Готово. Отчёты:"
find "$out" -maxdepth 1 -name '*.json' -newermt '-2 hours' | sort
