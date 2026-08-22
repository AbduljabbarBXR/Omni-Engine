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

def clean(rows):
    return "\n".join(re.sub(r"<[^>]+>", "", t) for t in rows)

ds2 = load_dataset("wikitext", "wikitext-2-raw-v1")
open("data/wikitext2_train.txt", "w", encoding="utf-8").write(clean(ds2["train"]["text"]))
open("data/wikitext2_valid.txt", "w", encoding="utf-8").write(clean(ds2["validation"]["text"]))

ds103 = load_dataset("wikitext", "wikitext-103-raw-v1")
open("data/wikitext103_slice_train.txt", "w", encoding="utf-8").write(clean(ds103["train"]["text"][:200000]))
open("data/wikitext103_slice_valid.txt", "w", encoding="utf-8").write(clean(ds103["validation"]["text"][:5000]))
print("data ready")
EOF
python -m omni.train --steps 2000 --batch-size 8 --seq-len 256 --data data/wikitext2_train.txt --base EleutherAI/pythia-70m-deduped --delta-penalty 0.1 --out runs/scale_70m --log-every 100
python -m omni.eval --run runs/scale_70m --data data/wikitext2_valid.txt --blocks 100
python -m omni.train --steps 2000 --batch-size 8 --seq-len 256 --data data/wikitext103_slice_train.txt --base EleutherAI/pythia-160m-deduped --delta-penalty 0.1 --out runs/scale_data --log-every 100
python -m omni.eval --run runs/scale_data --data data/wikitext103_slice_valid.txt --blocks 100
