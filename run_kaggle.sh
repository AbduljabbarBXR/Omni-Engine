#!/bin/bash
set -e
pip install -q transformers numpy
python -c "
import torch
ok = torch.cuda.is_available()
cap = torch.cuda.get_device_capability(0) if ok else (0, 0)
print('CUDA:', ok, torch.cuda.get_device_name(0) if ok else 'NO GPU', 'capability', cap)
if not ok or cap < (7, 0):
    raise SystemExit('GPU unsupported, rerun the push to get a different accelerator')
"
python stability_sweep.py --steps 120 --base EleutherAI/pythia-160m-deduped
python run_ablation.py --steps 2000 --base EleutherAI/pythia-160m-deduped --delta-penalty 0.1 --out runs/ablation2000
