#!/bin/bash
set -e
pip install -q transformers numpy
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python stability_sweep.py --steps 120 --base EleutherAI/pythia-160m-deduped
python run_ablation.py --steps 2000 --base EleutherAI/pythia-160m-deduped --delta-penalty 0.1 --out runs/ablation2000
