import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import load_base
from .config import Config
from .model import OmniModel


def sample(logits, temperature, top_k):
    if temperature <= 0:
        return logits.argmax(-1).item()
    probs = F.softmax(logits / temperature, dim=-1)
    if top_k > 0:
        topv, _ = torch.topk(probs, top_k)
        probs[probs < topv[-1]] = 0.0
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prompt", default="To be, or not to be")
    ap.add_argument("--tokens", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "cfg.json")
    base_model = args.base or cfg.base_model
    base, tokenizer = load_base(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)
    model = OmniModel(base, cfg).to(device)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    hebbian_path = run_dir / "hebbian.pt"
    if hebbian_path.exists() and model.hebbian is not None:
        banks = torch.load(hebbian_path, map_location="cpu")["banks"]
        for bank, E in zip(model.hebbian, banks):
            bank.E = E

    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"prompt: {args.prompt}", flush=True)
    generated = args.prompt
    with torch.no_grad():
        for _ in range(args.tokens):
            logits, _, _, _, _, _ = model(ids)
            next_id = sample(logits[0, -1], args.temperature, args.top_k)
            piece = tokenizer.decode([next_id])
            generated += piece
            print(piece, end="", flush=True)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    print()
    print(f"--- generated {args.tokens} tokens ---")
    print(generated, flush=True)


if __name__ == "__main__":
    main()
