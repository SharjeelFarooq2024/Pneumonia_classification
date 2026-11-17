"""Create the fixed stratified P06 training subsets.

Official setting (created once, reused by every experiment):
    python scripts/make_subset.py --fraction 0.25 --seed 42

Supplementary data-efficiency subsets:
    python scripts/make_subset.py --all-efficiency
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make `src` importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subset import create_subset, save_split_manifests, verify_subset  # noqa: E402
from src.utils import load_config, resolve_path, setup_logger  # noqa: E402


def _name_for(fraction: float, seed: int) -> str:
    return f"train{int(round(fraction * 100))}_seed{seed}.json"


def create_one(config, fraction: float, seed: int, force: bool, logger) -> dict:
    out = resolve_path(Path(config["data"]["splits_dir"]) / _name_for(fraction, seed))
    existed = out.exists()
    payload = create_subset(config, fraction=fraction, seed=seed, output_path=out, force=force)
    ok, report = verify_subset(payload)
    logger.info("%s %s\n%s", "Loaded existing" if existed and not force else "Created", out, report)
    if not ok:
        raise SystemExit("Subset class proportions deviate from the full train split by > 1 percentage point.")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create stratified train subsets.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Recreate an existing subset file.")
    parser.add_argument("--all-efficiency", action="store_true",
                        help="Create 10%, 25%, and 100% subsets (100% is supplementary context only).")
    parser.add_argument("--save-manifests", action="store_true",
                        help="Also save global indices and class counts for train/val/test.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    logger = setup_logger("make_subset")
    fractions = [0.10, 0.25, 1.00] if args.all_efficiency else [args.fraction]
    results = []
    for fraction in fractions:
        payload = create_one(config, fraction, args.seed, args.force, logger)
        results.append({
            "fraction": fraction,
            "n_subset": payload["n_subset"],
            "class_counts": payload["class_counts"],
            "max_abs_class_proportion_diff": payload["max_abs_class_proportion_diff"],
        })
    if args.save_manifests:
        manifest = save_split_manifests(config, config["data"]["splits_dir"])
        logger.info("Saved split manifests: %s", json.dumps({k: v["n"] for k, v in manifest.items()}))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
