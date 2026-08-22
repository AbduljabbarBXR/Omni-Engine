#!/bin/bash
set -e
pip install -q "numpy<2" "torch==2.2.2" "transformers==4.46.3" --extra-index-url https://download.pytorch.org/whl/cu118
python -c "
import torch
ok = torch.cuda.is_available()
arch = torch.cuda.get_arch_list() if ok else []
name = torch.cuda.get_device_name(0) if ok else 'NO GPU'
print('CUDA:', ok, name, arch)
if not ok:
    raise SystemExit('NO GPU')
if not any(a.startswith(('sm_6', 'sm_7', 'sm_8', 'sm_9')) for a in arch):
    raise SystemExit('torch build does not support this GPU')
"
python stability_sweep.py --steps 120 --base EleutherAI/pythia-160m-deduped
python run_ablation.py --steps 2000 --base EleutherAI/pythia-160m-deduped --delta-penalty 0.1 --out runs/ablation2000
