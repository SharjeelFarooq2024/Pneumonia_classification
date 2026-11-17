"""Reproducible command runner for the pneumonia project.

Run from the project root (or anywhere; the script changes to the project root):

    python scripts/run_experiments.py smoke
    python scripts/run_experiments.py main --seeds 42 43 44
    python scripts/run_experiments.py evaluate --seeds 42 43 44
    python scripts/run_experiments.py analysis
    python scripts/run_experiments.py ablations --seeds 42
    python scripts/run_experiments.py dataeff --seeds 42

The ``main`` command trains and evaluates both models and then runs the
analysis. It does not retrain a model whose checkpoint already exists unless
``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args, dry_run=False):
    printable = " ".join(str(a) for a in args)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(args, cwd=ROOT, check=True)


def train(config, seed, force=False, dry_run=False, extra=None):
    name = Path(config).stem
    checkpoint = ROOT / "checkpoints" / f"{name}_seed{seed}.pt"
    if checkpoint.exists() and not force:
        print(f"skip existing checkpoint: {checkpoint}")
        return
    args = [PY, "-m", "src.train", "--config", config, "--seed", str(seed)]
    if extra:
        args += extra
    run(args, dry_run=dry_run)


def evaluate(experiment, seed, force=False, dry_run=False, batch_size=None):
    checkpoint = ROOT / "checkpoints" / f"{experiment}_seed{seed}.pt"
    if not checkpoint.exists():
        print(f"missing checkpoint, skipping: {checkpoint}")
        return
    args = [PY, "-m", "src.evaluate", "--checkpoint", str(checkpoint), "--splits", "val", "test"]
    if force:
        args.append("--force")
    if batch_size:
        args += ["--batch-size", str(batch_size)]
    run(args, dry_run=dry_run)


def analysis(seed=42, dry_run=False):
    run([PY, "-m", "src.analysis", "--config", "configs/base.yaml", "aggregate"], dry_run)
    run([PY, "-m", "src.analysis", "--config", "configs/base.yaml", "--seed", str(seed), "figures"], dry_run)
    run([PY, "-m", "src.analysis", "--config", "configs/base.yaml", "--seed", str(seed), "bootstrap"], dry_run)
    run([PY, "-m", "src.analysis", "--config", "configs/base.yaml", "--seed", str(seed), "gradcam"], dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "smoke", "main", "baseline", "proposed", "evaluate", "analysis",
        "ablations", "dataeff",
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--force", action="store_true", help="Retrain/overwrite existing checkpoints.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Override evaluation batch size (useful on low-memory machines).")
    args = parser.parse_args(argv)

    if args.command == "smoke":
        run([PY, "-m", "src.train", "--config", "configs/baseline.yaml", "--seed", "42", "--smoke"], args.dry_run)
        run([PY, "-m", "src.train", "--config", "configs/proposed.yaml", "--seed", "42", "--smoke"], args.dry_run)
        return 0

    if args.command in ("main", "baseline"):
        for seed in args.seeds:
            train("configs/baseline.yaml", seed, args.force, args.dry_run)
        if args.command == "main":
            for seed in args.seeds:
                evaluate("baseline", seed, args.force, args.dry_run, args.eval_batch_size)

    if args.command in ("main", "proposed"):
        for seed in args.seeds:
            train("configs/proposed.yaml", seed, args.force, args.dry_run)
        if args.command == "main":
            for seed in args.seeds:
                evaluate("proposed", seed, args.force, args.dry_run, args.eval_batch_size)

    if args.command == "evaluate":
        for seed in args.seeds:
            evaluate("baseline", seed, args.force, args.dry_run, args.eval_batch_size)
            evaluate("proposed", seed, args.force, args.dry_run, args.eval_batch_size)
        return 0

    if args.command == "analysis":
        analysis(seed=args.seeds[0], dry_run=args.dry_run)
        return 0

    if args.command == "ablations":
        configs = sorted((ROOT / "configs" / "ablations").glob("a*.yaml"))
        for seed in args.seeds:
            for config in configs:
                train(f"configs/ablations/{config.name}", seed, args.force, args.dry_run,
                      extra=["--set", "training.batch_size=16", "eval.batch_size=16", "data.num_workers=0"])
        return 0

    if args.command == "dataeff":
        configs = sorted((ROOT / "configs" / "ablations").glob("dataeff_*.yaml"))
        for seed in args.seeds:
            for config in configs:
                train(f"configs/ablations/{config.name}", seed, args.force, args.dry_run,
                      extra=["--set", "training.batch_size=16", "eval.batch_size=16", "data.num_workers=0"])
        return 0

    if args.command == "main":
        analysis(seed=args.seeds[0], dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

