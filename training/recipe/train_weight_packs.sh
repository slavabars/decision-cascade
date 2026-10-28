#!/usr/bin/env bash
# Обучает матрицу паков весов: {support,public,routerbench} x
# {gpt2,distilgpt2,gpt2-medium}.
#
# Каждый пак — отдельный каталог $PACKS_ROOT/<dataset>-<label>/ с heads.pt,
# копией под подписью вопросов (drop-in для DECISION_CHECKPOINT_DIR) и
# metadata.json. Веса не коммитим в git — публикуем релизом
# (training/recipe/publish_weight_packs.sh).
#
# Переопределяется окружением: PACKS_ROOT, PYTHON, DECISION_DEVICE,
# DECISION_DTYPE, SUPPORT_N/STEPS/RLCD, PUBLIC_LIMIT/STEPS/RLCD,
# ROUTERBENCH_LIMIT/STEPS/RLCD/SHOT.
set -euo pipefail

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" && -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi
python_bin="${python_bin:-python}"
packs_root="${PACKS_ROOT:-checkpoints/packs}"
device="${DECISION_DEVICE:-auto}"
dtype="${DECISION_DTYPE:-float32}"

support_n="${SUPPORT_N:-1000}"
support_steps="${SUPPORT_STEPS:-500}"
support_rlcd="${SUPPORT_RLCD:-300}"

public_limit="${PUBLIC_LIMIT:-4000}"
public_steps="${PUBLIC_STEPS:-500}"
public_rlcd="${PUBLIC_RLCD:-0}"

routerbench_limit="${ROUTERBENCH_LIMIT:-4000}"
routerbench_steps="${ROUTERBENCH_STEPS:-500}"
routerbench_rlcd="${ROUTERBENCH_RLCD:-200}"
routerbench_shot="${ROUTERBENCH_SHOT:-0shot}"

# "<label>:<encoder>:<dim>"
encoders=(
  "gpt2:openai-community/gpt2:768"
  "distilgpt2:distilgpt2:768"
  "gpt2-medium:openai-community/gpt2-medium:1024"
)

train_pack() {
  local dataset="$1" label="$2" encoder="$3" dim="$4" n="$5" steps="$6" rlcd="$7"
  local out="$packs_root/${dataset}-${label}"
  echo
  echo "▶ $dataset / $label ($encoder, dim=$dim, n=$n, steps=$steps, rlcd=$rlcd)"
  local args=(--specs "$dataset" --out-dir "$out" --encoder "$encoder"
              --dim "$dim" --train-steps "$steps" --rlcd-steps "$rlcd" --seed 0)
  if [[ "$dataset" == "support" ]]; then
    args+=(--generate "$n")
  else
    args+=(--limit "$n")
  fi
  if [[ "$dataset" == "routerbench" ]]; then
    args+=(--shot "$routerbench_shot")
  fi
  DECISION_DEVICE="$device" DECISION_DTYPE="$dtype" \
    "$python_bin" -m training.recipe.train_heads "${args[@]}"
}

for spec in "${encoders[@]}"; do
  IFS=: read -r label encoder dim <<< "$spec"
  train_pack support "$label" "$encoder" "$dim" \
    "$support_n" "$support_steps" "$support_rlcd"
  train_pack public "$label" "$encoder" "$dim" \
    "$public_limit" "$public_steps" "$public_rlcd"
  train_pack routerbench "$label" "$encoder" "$dim" \
    "$routerbench_limit" "$routerbench_steps" "$routerbench_rlcd"
done

echo
echo "Готово. Паки:"
find "$packs_root" -maxdepth 1 -mindepth 1 -type d | sort
