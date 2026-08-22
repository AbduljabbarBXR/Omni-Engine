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
python -m omni.train --steps 2000 --batch-size 8 --seq-len 256 --data data/wikitext2_train.txt --base EleutherAI/pythia-160m-deduped --delta-penalty 0.1 --out-harness --out-bridge --out runs/slice2_wiki --log-every 100
python -m omni.eval --run runs/slice2_wiki --data data/wikitext2_valid.txt --blocks 100
python -m omni.train --steps 2000 --batch-size 8 --seq-len 256 --data data/tinyshakespeare.txt --base EleutherAI/pythia-160m-deduped --delta-penalty 0.1 --out-harness --out-bridge --out runs/slice2_shake --log-every 100
python -m omni.eval --run runs/slice2_shake --data data/tinyshakespeare.txt --blocks 100
