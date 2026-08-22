import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .base import load_base
from .config import Config
from .dataset import TextDataset
from .memory import MemoryStore
from .model import OmniModel


def save_checkpoint(model, out_dir, cfg):
    keys = ["expert_layers", "message_passing", "output_harness"]
    state = {k: v.cpu() for k, v in model.state_dict().items() if k.startswith(tuple(keys))}
    torch.save(state, out_dir / "model.pt")
    if model.hebbian is not None:
        torch.save({"banks": [b.E for b in model.hebbian]}, out_dir / "hebbian.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/omni-160m")
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--base", default="models/pythia-160m-deduped")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--n-expert-layers", type=int, default=4)
    ap.add_argument("--mid-dim", type=int, default=256)
    ap.add_argument("--graph-rounds", type=int, default=0)
    ap.add_argument("--edge-mode", default="learned")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-harness", action="store_true")
    ap.add_argument("--out-bridge", action="store_true")
    ap.add_argument("--out-top-k", type=int, default=64)
    ap.add_argument("--n-out-experts", type=int, default=4)
    ap.add_argument("--out-expert-top-k", type=int, default=1)
    ap.add_argument("--hebbian-lr", type=float, default=0.0)
    ap.add_argument("--delta-scale", type=float, default=0.02)
    ap.add_argument("--delta-penalty", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    cfg = Config(
        base_model=args.base,
        n_experts=args.n_experts,
        top_k=args.top_k,
        n_expert_layers=args.n_expert_layers,
        mid_dim=args.mid_dim,
        graph_rounds=args.graph_rounds,
        edge_mode=args.edge_mode,
        out_harness=args.out_harness,
        out_bridge=args.out_bridge,
        out_top_k=args.out_top_k,
        n_out_experts=args.n_out_experts,
        out_expert_top_k=args.out_expert_top_k,
        seed=args.seed,
        hebbian_lr=args.hebbian_lr,
        delta_scale=args.delta_scale,
        delta_penalty=args.delta_penalty,
        weight_decay=args.weight_decay,
        aux_coef=args.aux_coef,
        lr=args.lr,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "cfg.json")

    base, tokenizer = load_base(args.base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)
    model = OmniModel(base, cfg).to(device)
    model.base.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train / 1e6:.2f}M", flush=True)

    with open(args.data, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    full_ids = tokenizer(text, return_tensors="pt", truncation=True)["input_ids"][0]
    n_tokens = full_ids.numel()
    n_train_tokens = int(n_tokens * 0.95)
    train_ids = full_ids[:n_train_tokens]
    val_ids = full_ids[n_train_tokens:]

    train_ds = TextDataset(train_ids, cfg.seq_len)
    val_ds = TextDataset(val_ids, cfg.seq_len)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    print(f"tokens: {n_tokens} train blocks: {len(train_ds)} val blocks: {len(val_ds)}", flush=True)

    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    store = MemoryStore(out_dir / "omni_memory.db")

    def eval_ppl(dataset, max_blocks=64):
        model.eval()
        dloader = DataLoader(dataset, batch_size=cfg.batch_size)
        total_nll = 0.0
        total_tok = 0
        count = 0
        for batch in dloader:
            if count >= max_blocks:
                break
            with torch.no_grad():
                _, loss, _, _, _, _ = model(batch.to(device), batch.to(device))
            total_nll += loss.item() * batch.size(0) * (batch.size(1) - 1)
            total_tok += batch.size(0) * (batch.size(1) - 1)
            count += 1
        model.train()
        if total_tok == 0:
            return float("nan")
        return float(np.exp(total_nll / total_tok))

    base.eval()
    with torch.no_grad():
        vblock = val_ds[0].unsqueeze(0).to(device)
        base_ppl = float(np.exp(
            base(vblock, labels=vblock).loss.item()
        ))
    print(f"frozen base ppl (1 block): {base_ppl:.2f}", flush=True)

    model.train()
    step = 0
    epoch = 0
    t0 = time.time()
    while step < args.steps:
        epoch += 1
        perm = torch.randperm(len(train_ds))
        for i in perm:
            if step >= args.steps:
                break
            idx = torch.randint(0, len(train_ds), (cfg.batch_size,))
            batch = torch.stack([train_ds[int(j)] for j in idx]).to(device)
            logits, loss, aux, delta_sq, usages, ifls = model(batch, batch)
            total = loss + cfg.aux_coef * aux + cfg.delta_penalty * delta_sq
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            if model.hebbian is not None:
                for bank, ifl in zip(model.hebbian, ifls):
                    bank.update(ifl.detach().cpu())
            step += 1
            if step % args.log_every == 0:
                secs = (time.time() - t0) / step
                print(f"step {step:5d} loss {loss.item():.3f} aux {aux.item():.3f} "
                      f"delta {delta_sq.item():.4f} usage {np.mean(usages):.2f} {secs:.1f}s/step",
                      flush=True,
                )
            if step % 100 == 0:
                ppl = eval_ppl(val_ds)
                print(f"step {step:5d} val ppl {ppl:.2f}", flush=True)
                model.train()
            if step >= args.steps:
                break

    save_checkpoint(model, out_dir, cfg)
    for i, block in enumerate(model.expert_layers):
        for j, ex in enumerate(block.experts):
            store.save_expert(i, j, ex.state_dict())
    print(f"checkpoint saved -> {out_dir}", flush=True)
    print(f"sqlite: {store.expert_count()} expert weights stored", flush=True)


if __name__ == "__main__":
    main()
