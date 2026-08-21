import argparse
import subprocess
import sys
from pathlib import Path

CONFIGS = [
    ("scale005_lr3e4", ["--delta-scale", "0.05"]),
    ("scale002_lr1e4", ["--delta-scale", "0.02", "--lr", "1e-4"]),
    ("scale010_lr3e4", ["--delta-scale", "0.10"]),
    ("scale005_lr3e5", ["--delta-scale", "0.05", "--lr", "3e-5"]),
    ("scale005_aux01", ["--delta-scale", "0.05", "--aux-coef", "0.1"]),
]


def parse_metrics(output):
    base_ppl = None
    last_loss = None
    for line in output.splitlines():
        if line.startswith("frozen base ppl"):
            base_ppl = float(line.split(":")[1].strip())
        if line.startswith("step"):
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "loss":
                    last_loss = float(parts[i + 1])
    return base_ppl, last_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--base", default="models/pythia-160m-deduped")
    ap.add_argument("--out", default="runs/sweep")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, extra in CONFIGS:
        out_dir = out_root / name
        cmd = [
            sys.executable, "-m", "omni.train",
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--data", args.data,
            "--base", args.base,
            "--out", str(out_dir),
            "--log-every", "20",
        ] + extra
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-1500:])
            print(proc.stderr[-1500:])
            rows.append((name, None, None))
            continue
        base_ppl, last_loss = parse_metrics(proc.stdout)
        rows.append((name, base_ppl, last_loss))

    print()
    print("=" * 46)
    print(f"{'config':<18}{'base ppl':>10}{'final loss':>12}")
    print("=" * 46)
    for name, base_ppl, last_loss in rows:
        bp = f"{base_ppl:.2f}" if base_ppl else "failed"
        fl = f"{last_loss:.3f}" if last_loss else "failed"
        print(f"{name:<18}{bp:>10}{fl:>12}")
    print("=" * 46)
    print("final loss below base nll means the model is improving.")


if __name__ == "__main__":
    main()
