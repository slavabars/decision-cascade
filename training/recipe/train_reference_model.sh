#!/usr/bin/env bash
# Обучает референс-модель голов: gpt2-энкодер + SFT + RLCD-калибровка noul.
#
# На выходе каталог с heads.pt, копией под подписью вопросов и metadata.json
# (метрики train/val/test). Каталог можно подать сервису как
# DECISION_CHECKPOINT_DIR — тогда веса подхватятся автоматически.
set -euo pipefail

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" && -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi
python_bin="${python_bin:-python}"
out_dir="${OUT_DIR:-checkpoints/reference}"
encoder="${ENCODER:-openai-community/gpt2}"
dim="${DIM:-768}"
n="${N:-1000}"
train_steps="${TRAIN_STEPS:-300}"
rlcd_steps="${RLCD_STEPS:-200}"
seed="${SEED:-0}"
device="${DECISION_DEVICE:-auto}"
dtype="${DECISION_DTYPE:-float32}"

DECISION_DEVICE="$device" DECISION_DTYPE="$dtype" "$python_bin" -m training.recipe.train_heads \
  --generate "$n" \
  --out-dir "$out_dir" \
  --encoder "$encoder" \
  --dim "$dim" \
  --train-steps "$train_steps" \
  --rlcd-steps "$rlcd_steps" \
  --seed "$seed"

echo
echo "Готово. Drop-in для сервиса:"
echo "  DECISION_CHECKPOINT_DIR=$out_dir \\"
echo "  DECISION_MODEL_ID=$encoder DECISION_FEATURE_DIM=$dim \\"
echo "  uvicorn api.server.main:app"
