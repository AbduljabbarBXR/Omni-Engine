import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import load_base
from .config import Config
from .dataset import TextDataset
from .memory import MemoryStore
from .model import OmniModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/omni-160m")
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--blocks", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "cfg.json")
    base, tokenizer = load_base(cfg.base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)
    model = OmniModel(base, cfg).to(device)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    hebbian_path = run_dir / "hebbian.pt"
    if hebbian_path.exists() and model.hebbian is not None:
        banks = torch.load(hebbian_path, map_location="cpu")["banks"]
        for bank, E in zip(model.hebbian, banks):
            bank.E = E

    with open(args.data, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    full_ids = tokenizer(text, return_tensors="pt", truncation=True)["input_ids"][0]
    n_train = int(full_ids.numel() * 0.95)
    val_ds = TextDataset(full_ids[n_train:], cfg.seq_len)
    loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model.eval()
    base.eval()
    total_omni = 0.0
    total_base = 0.0
    total_tok = 0
    count = 0
    for batch in loader:
        if count >= args.blocks:
            break
        with torch.no_grad():
            _, loss_omni, _, _, _, _ = model(batch.to(device), batch.to(device))
            loss_base = base(batch.to(device), labels=batch.to(device)).loss
        n = batch.size(0) * (batch.size(1) - 1)
        total_omni += loss_omni.item() * n
        total_base += loss_base.item() * n
        total_tok += n
        count += 1

    print(f"frozen base ppl: {np.exp(total_base / total_tok):.2f}")
    print(f"omni      ppl:   {np.exp(total_omni / total_tok):.2f}")

    store = MemoryStore(run_dir / "omni_memory.db")
    with torch.no_grad():
        for i in range(min(5, len(val_ds))):
            block = val_ds[i].unsqueeze(0).to(device)
            out = base(block, output_hidden_states=True)
            emb = out.hidden_states[-1][0, -1]
            txt = tokenizer.decode(block[0].tolist())[:200]
            store.store_memory(emb, txt)
    print(f"sqlite experts: {store.expert_count()} memories: {store.memory_count()}")

    if model.hebbian is not None:
        for i, bank in enumerate(model.hebbian):
            order = torch.argsort(bank.E.flatten(), descending=True)[:3]
            top = [(int(o // bank.n), int(o % bank.n), float(bank.E.flatten()[o])) for o in order]
            print(f"hebbian layer {i} strongest edges: {top}")

    probe = tokenizer(
        "To be, or not to be, that is the question", return_tensors="pt"
    )["input_ids"].to(device)
    with torch.no_grad():
        out = base(probe, output_hidden_states=True)
        emb = out.hidden_states[-1][0, -1]
    hits = store.query_memories(emb, k=3)
    print("memory probe hits:")
    for hid, score, txt in hits:
        print(f"  {score:.2f} {txt[:60]!r}")


if __name__ == "__main__":
    main()
