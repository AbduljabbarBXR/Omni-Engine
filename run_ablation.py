import argparse
import subprocess
import sys
from pathlib import Path

CONFIGS = {
    "flat": [],
    "graph": ["--graph-rounds", "1"],
    "hebbian": ["--hebbian-lr", "0.05"],
    "full": ["--graph-rounds", "1", "--hebbian-lr", "0.05"],
}


def run(cmd):
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        sys.exit(1)
    return proc.stdout


def parse_ppl(output):
    base_ppl = omni_ppl = None
    for line in output.splitlines():
        if line.startswith("frozen base ppl"):
            base_ppl = float(line.split(":")[1].strip())
        if line.startswith("omni      ppl"):
            omni_ppl = float(line.split(":")[1].strip())
    return base_ppl, omni_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--eval-data", default=None)
    ap.add_argument("--base", default="models/pythia-160m-deduped")
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--runs", default="flat,graph,hebbian,full")
    ap.add_argument("--eval-blocks", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--delta-scale", type=float, default=None)
    ap.add_argument("--delta-penalty", type=float, default=None)
    ap.add_argument("--aux-coef", type=float, default=None)
    args = ap.parse_args()

    extra = []
    if args.lr is not None:
        extra += ["--lr", str(args.lr)]
    if args.delta_scale is not None:
        extra += ["--delta-scale", str(args.delta_scale)]
    if args.delta_penalty is not None:
        extra += ["--delta-penalty", str(args.delta_penalty)]
    if args.aux_coef is not None:
        extra += ["--aux-coef", str(args.aux_coef)]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.runs.split(","):
        name = name.strip()
        if name not in CONFIGS:
            sys.exit(f"unknown run: {name}")
        out_dir = out_root / name
        cmd = [
            sys.executable, "-m", "omni.train",
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--data", args.data,
            "--base", args.base,
            "--out", str(out_dir),
            "--log-every", "50",
        ] + CONFIGS[name] + extra
        output = run(cmd)
        loss_lines = [l.strip() for l in output.splitlines() if " loss " in l]
        val_lines = [l.strip() for l in output.splitlines() if "val ppl" in l]
        print(f"[{name}] last train: {loss_lines[-1] if loss_lines else 'n/a'}")
        print(f"[{name}] val ppl trajectory: {', '.join(v.split()[-1] for v in val_lines[-5:])}")
        if out_dir.joinpath("model.pt").exists():
            eval_data = args.eval_data or args.data
            output = run([
                sys.executable, "-m", "omni.eval",
                "--run", str(out_dir),
                "--data", eval_data,
                "--blocks", str(args.eval_blocks),
            ])
        base_ppl, omni_ppl = parse_ppl(output)
        results.append((name, base_ppl, omni_ppl))

    print()
    print("=" * 52)
    print(f"{'run':<10}{'base ppl':>12}{'omni ppl':>12}{'delta':>10}")
    print("=" * 52)
    for name, base_ppl, omni_ppl in results:
        delta = omni_ppl - base_ppl if omni_ppl and base_ppl else float("nan")
        print(f"{name:<10}{base_ppl:>12.2f}{omni_ppl:>12.2f}{delta:>10.2f}")
    print("=" * 52)
    print("delta < 0 means Omni beats the frozen base.")


if __name__ == "__main__":
    main()
