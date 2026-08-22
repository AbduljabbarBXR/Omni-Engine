#!/bin/bash
set -e
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
pip install -q "numpy<2" "torch==2.2.2" "transformers==4.44.2" --extra-index-url https://download.pytorch.org/whl/cu118
python -c "
import torch
ok = torch.cuda.is_available()
print('CUDA:', ok, torch.cuda.get_device_name(0) if ok else 'NO GPU')
assert ok, 'no gpu'
"
python - <<'EOF'
import re
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
def clean(rows):
    return "\n".join(re.sub(r"<[^>]+>", "", t) for t in rows)
open("data/wikitext2_train.txt", "w", encoding="utf-8").write(clean(ds["train"]["text"]))
open("data/wikitext2_valid.txt", "w", encoding="utf-8").write(clean(ds["validation"]["text"]))
print("wikitext2 ready")
EOF

run_one () {
  local name=$1
  shift
  local out
  out=$(python -m omni.train --steps 120 --batch-size 8 --seq-len 256 \
    --data data/wikitext2_train.txt --base Qwen/Qwen2.5-0.5B \
    --out runs/sweep05/$name --log-every 40 "$@" 2>&1)
  local base loss
  base=$(echo "$out" | grep -oP 'frozen base ppl \(1 block\): \K[0-9.]+' | tail -1)
  loss=$(echo "$out" | grep -oP 'step\s+\d+ loss \K[0-9.]+' | tail -1)
  echo "$name base_ppl=${base:-failed} final_loss=${loss:-failed}"
}

run_one scale002_lr1e4_pen01 --delta-scale 0.02 --lr 1e-4 --delta-penalty 0.1
run_one scale002_lr1e4_pen1 --delta-scale 0.02 --lr 1e-4 --delta-penalty 1.0
run_one scale005_lr1e4_pen01 --delta-scale 0.05 --lr 1e-4 --delta-penalty 0.1
run_one scale002_lr3e4_pen01 --delta-scale 0.02 --lr 3e-4 --delta-penalty 0.1
run_one scale002_lr5e5_pen01 --delta-scale 0.02 --lr 5e-5 --delta-penalty 0.1

echo "SWEEP05_DONE"
