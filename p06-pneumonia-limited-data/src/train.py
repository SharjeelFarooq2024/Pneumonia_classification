"""Shared training entry point for every experiment.

Usage:
    python -m src.train --config configs/baseline.yaml --seed 42
    python -m src.train --config configs/proposed.yaml --seed 42

The test split is never loaded here. Model selection uses validation macro-F1.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.data.dataset import (
    load_subset_positions,
    make_dataloader,
    make_datasets,
)
from src.models.proposed import (
    build_model,
    build_optimizer,
    build_scheduler,
    set_trainable,
    trainable_summary,
)
from src.utils import (
    amp_is_available,
    compute_metrics,
    experiment_results_dir,
    count_parameters,
    dump_environment,
    environment_info,
    get_device,
    git_commit,
    load_config,
    predict_loader,
    probe_batch_size,
    resolve_path,
    save_json,
    set_seed,
    setup_logger,
    to_jsonable,
)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal = -((1.0 - pt) ** self.gamma) * log_pt
        if self.label_smoothing > 0:
            smooth = -log_prob.mean(dim=1)
            focal = (1.0 - self.label_smoothing) * focal + self.label_smoothing * smooth
        if self.weight is not None:
            focal = focal * self.weight[target]
        return focal.mean()


def build_criterion(config: Mapping[str, Any], class_weights: Optional[torch.Tensor],
                    device: torch.device) -> nn.Module:
    training = config["training"]
    name = str(training.get("loss", "ce")).lower()
    ls = float(training.get("label_smoothing", 0.0))
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    if name == "weighted_ce":
        if class_weights is None:
            raise ValueError("weighted_ce requires class weights from the training subset.")
        return nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=ls)
    if name == "focal":
        weight = class_weights if training.get("class_weights_from_subset", False) else None
        return FocalLoss(gamma=float(training.get("focal_gamma", 2.0)), weight=weight, label_smoothing=ls)
    raise ValueError(f"Unknown loss: {name}")


def class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def mixup_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float,
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=device)
    mixed = lam * images + (1.0 - lam) * images[perm]
    return mixed, targets, targets[perm], lam


def cutmix_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float,
                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=device)
    _, _, h, w = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, w)
    y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, h)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    lam_effective = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    return mixed, targets, targets[perm], float(lam_effective)


def run_train_epoch(model, loader, criterion, optimizer, scaler, device, amp,
                    grad_clip, mixup_alpha, cutmix_alpha, limit_batches=0) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_idx, (images, targets, _) in enumerate(tqdm(loader, desc="train", leave=False, disable=not sys.stderr.isatty())):
        if limit_batches and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mix_y_a = mix_y_b = None
        lam = 1.0
        if cutmix_alpha and cutmix_alpha > 0:
            images, mix_y_a, mix_y_b, lam = cutmix_batch(images, targets, cutmix_alpha, device)
        elif mixup_alpha and mixup_alpha > 0:
            images, mix_y_a, mix_y_b, lam = mixup_batch(images, targets, mixup_alpha, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            if mix_y_a is not None:
                loss = lam * criterion(logits, mix_y_a) + (1.0 - lam) * criterion(logits, mix_y_b)
            else:
                loss = criterion(logits, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += batch_size
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "n": total_seen,
    }


@torch.no_grad()
def evaluate_split(model, loader, criterion, device, amp, class_names,
                   bootstrap_n: int = 0, bootstrap_seed: int = 42,
                   limit_batches: int = 0) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    true_all: List[np.ndarray] = []
    prob_all: List[np.ndarray] = []
    idx_all: List[np.ndarray] = []
    for batch_idx, (images, targets, indices) in enumerate(tqdm(loader, desc="val", leave=False, disable=not sys.stderr.isatty())):
        if limit_batches and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets_dev = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets_dev)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        total_loss += float(loss.item()) * images.size(0)
        total_seen += images.size(0)
        true_all.append(targets.numpy())
        prob_all.append(probs)
        idx_all.append(indices.numpy())
    y_true = np.concatenate(true_all) if true_all else np.array([], dtype=int)
    y_prob = np.concatenate(prob_all) if prob_all else np.zeros((0, len(class_names)))
    indices = np.concatenate(idx_all) if idx_all else np.array([], dtype=int)
    metrics = compute_metrics(y_true, y_prob, class_names, bootstrap_n=bootstrap_n,
                              bootstrap_seed=bootstrap_seed)
    metrics["loss"] = total_loss / max(total_seen, 1)
    return metrics, y_true, y_prob, indices


def _stage_list(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    training = config["training"]
    stages = training.get("stages")
    if not stages:
        stages = [{
            "name": "single",
            "epochs": int(training.get("epochs", 20)),
            "lr": float(training.get("lr", 1e-3)),
            "weight_decay": float(training.get("weight_decay", 0.0)),
            "unfreeze": "all",
            "scheduler": "none",
        }]
    return [dict(s) for s in stages]


def train(config: Mapping[str, Any], seed: int, smoke: bool = False,
          limit_batches: int = 0, device_override: Optional[str] = None) -> Dict[str, Any]:
    config = copy.deepcopy(dict(config))
    config["seed"] = int(seed)
    exp = str(config.get("experiment", "run"))
    if smoke:
        exp = f"{exp}_smoke"
        config["experiment"] = exp
        config.setdefault("data", {})["num_workers"] = 0
        smoke_stages = _stage_list(config)
        for stage in smoke_stages:
            stage["epochs"] = 1
        config["training"]["stages"] = smoke_stages
        config["training"]["batch_size"] = min(int(config["training"].get("batch_size", 8)), 8)
        limit_batches = limit_batches or 2

    run_dir = experiment_results_dir(config, exp) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = resolve_path(Path(config["paths"]["logs"]) / "runs" / f"{exp}_seed{seed}")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"train.{exp}.{seed}", run_dir / "train.log",
                          level=config.get("logging", {}).get("level", "INFO"))

    set_seed(seed)
    device = get_device(device_override or config.get("device", "auto"))
    env = dump_environment(config, device, log_dir="logs")
    logger.info("Experiment=%s seed=%d device=%s", exp, seed, device)
    logger.info("Versions: %s", json.dumps(env["versions"]))
    logger.info("Config: %s", json.dumps(to_jsonable(config), sort_keys=True))

    # Save exact config with every run.
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(to_jsonable(config), f, sort_keys=False)
    with open(log_dir / "config.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(to_jsonable(config), f, sort_keys=False)

    subset_file = config["data"]["subset_file"]
    positions = load_subset_positions(subset_file)
    with open(resolve_path(subset_file), "r", encoding="utf-8") as f:
        subset_payload = json.load(f)
    logger.info("Training subset: %s (%d samples, fraction %.2f, seed %s)",
                subset_file, len(positions), subset_payload.get("fraction", float("nan")),
                subset_payload.get("seed"))

    train_ds = make_datasets(config, "train", positions=positions, train=True)
    val_ds = make_datasets(config, "val", positions=None, train=False)
    logger.info("Dataset sizes: train=%d val=%d (test is not loaded in train.py)",
                len(train_ds), len(val_ds))
    logger.info("Train class counts: %s", np.bincount(train_ds.labels, minlength=config["data"]["num_classes"]).tolist())
    logger.info("Val class counts: %s", np.bincount(val_ds.labels, minlength=config["data"]["num_classes"]).tolist())

    model = build_model(config)
    model.to(device)
    batch_size = probe_batch_size(model, config, device, logger)
    gc.collect()
    train_loader = make_dataloader(train_ds, config, train=True, batch_size=batch_size, seed=seed)
    val_loader = make_dataloader(val_ds, config, train=False,
                                 batch_size=int(config["eval"].get("batch_size", 64)), seed=seed)

    class_weights = None
    if bool(config["training"].get("class_weights_from_subset", False)):
        class_weights = class_weights_from_labels(train_ds.labels, int(config["data"]["num_classes"]))
        logger.info("Class weights from train subset: %s", class_weights.tolist())
    criterion = build_criterion(config, class_weights, device)

    amp = amp_is_available(config, device)
    scaler = torch.amp.GradScaler(device="cuda") if (amp and device.type == "cuda") else None
    logger.info("Mixed precision: %s", amp)
    logger.info("Model parameters: %s", trainable_summary(model))

    history_rows: List[Dict[str, Any]] = []
    best = {"macro_f1": -1.0, "epoch": -1, "stage": None, "val_loss": float("inf")}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    global_epoch = 0
    start_time = time.time()
    stages = _stage_list(config)
    logger.info("Training schedule: %s", json.dumps([{k: v for k, v in s.items() if k != "layer_lr_mult"} for s in stages]))

    for stage_idx, stage in enumerate(stages):
        gc.collect()
        stage_name = stage.get("name", f"stage{stage_idx}")
        stage_epochs = int(stage.get("epochs", config["training"].get("epochs", 20)))
        set_trainable(model, stage.get("unfreeze", "all"))
        lr = float(stage.get("lr", config["training"].get("lr", 1e-3)))
        wd = float(stage.get("weight_decay", config["training"].get("weight_decay", 0.0)))
        optimizer = build_optimizer(
            model, lr=lr, weight_decay=wd,
            layer_lr_mult=stage.get("layer_lr_mult"),
            optimizer_name=str(config["training"].get("optimizer", "adam")),
        )
        scheduler = build_scheduler(optimizer, stage.get("scheduler", "none"), stage_epochs)
        logger.info("Stage %d/%d '%s': epochs=%d lr=%g wd=%g trainable=%s",
                    stage_idx + 1, len(stages), stage_name, stage_epochs, lr, wd,
                    trainable_summary(model)["trainable_parameters"])

        for epoch_in_stage in range(stage_epochs):
            epoch_start = time.time()
            train_metrics = run_train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, amp,
                config["training"].get("grad_clip"),
                float(config["training"].get("mixup_alpha", 0.0)),
                float(config["training"].get("cutmix_alpha", 0.0)),
                limit_batches=limit_batches,
            )
            val_metrics, _, _, _ = evaluate_split(
                model, val_loader, criterion, device, amp, config["data"]["class_names"],
                bootstrap_n=0, limit_batches=limit_batches,
            )
            if scheduler is not None:
                scheduler.step()
            global_epoch += 1
            row = {
                "epoch": global_epoch,
                "stage": stage_name,
                "epoch_in_stage": epoch_in_stage + 1,
                "lr_head": max(g["lr"] for g in optimizer.param_groups),
                "lr_min": min(g["lr"] for g in optimizer.param_groups),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_macro_auroc": val_metrics["macro_auroc"],
                "epoch_seconds": time.time() - epoch_start,
            }
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)
            logger.info(
                "epoch %d | %s | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f bal_acc %.4f macro_f1 %.4f auroc %.4f | %.1fs",
                global_epoch, stage_name, train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["balanced_accuracy"],
                val_metrics["macro_f1"], val_metrics["macro_auroc"], row["epoch_seconds"],
            )
            if val_metrics["macro_f1"] > best["macro_f1"]:
                best = {
                    "macro_f1": float(val_metrics["macro_f1"]),
                    "epoch": global_epoch,
                    "stage": stage_name,
                    "val_loss": float(val_metrics["loss"]),
                }
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                logger.info("New best val macro-F1 %.4f at epoch %d; checkpoint state updated.",
                            best["macro_f1"], global_epoch)

        del optimizer
        if scheduler is not None:
            del scheduler
        gc.collect()

    wall_seconds = time.time() - start_time
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    checkpoint = {
        "model_state": best_state,
        "config": to_jsonable(config),
        "seed": int(seed),
        "experiment": exp,
        "best": best,
        "class_names": list(config["data"]["class_names"]),
        "subset_file": subset_file,
        "input_channels": int(config["model"].get("input_channels", 3)),
        "model_name": str(config["model"].get("name")),
        "imagenet_normalization": bool(config["model"].get("imagenet_normalization", False)),
    }
    ckpt_path = resolve_path(Path(config["paths"]["checkpoints"]) / f"{exp}_seed{seed}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, ckpt_path)
    canonical = resolve_path(Path(config["paths"]["checkpoints"]) / f"{exp}.pt")
    if int(seed) == int(config.get("seed", 42)):
        torch.save(checkpoint, canonical)
    logger.info("Saved best checkpoint to %s (best val macro-F1 %.4f)", ckpt_path, best["macro_f1"])

    # Recompute validation metrics from the best checkpoint (bootstrap CI on val only).
    model.load_state_dict(best_state)
    val_metrics, y_true, y_prob, indices = evaluate_split(
        model, val_loader, criterion, device, amp, config["data"]["class_names"],
        bootstrap_n=int(config["eval"].get("bootstrap_n", 1000)),
        bootstrap_seed=int(config["eval"].get("bootstrap_seed", 42)),
    )
    save_json(val_metrics, run_dir / "val_metrics.json")
    from src.utils import save_predictions_csv
    save_predictions_csv(run_dir / "predictions_val.csv", y_true, y_prob, indices,
                         config["data"]["class_names"])

    run_info = {
        "experiment": exp,
        "seed": int(seed),
        "device": str(device),
        "batch_size": int(batch_size),
        "amp": bool(amp),
        "wall_clock_seconds": float(wall_seconds),
        "epochs_run": int(global_epoch),
        "best": best,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "train_subset": subset_payload,
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "checkpoint": str(ckpt_path),
        "config_path": str(run_dir / "config.yaml"),
        "versions": environment_info(device)["versions"],
        "git_commit": git_commit(),
    }
    save_json(run_info, run_dir / "run_info.json")
    logger.info("Run complete in %.1f s (%.1f min).", wall_seconds, wall_seconds / 60.0)
    return {"run_dir": str(run_dir), "checkpoint": str(ckpt_path), "best": best,
            "wall_clock_seconds": wall_seconds, "run_info": run_info}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a pneumonia classification model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config (relative to project root).")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--smoke", action="store_true", help="Tiny 2-batch, 1-epoch-per-stage smoke run.")
    parser.add_argument("--limit-batches", type=int, default=0, help="Limit batches per epoch (0 = all).")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect).")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None,
                        help="Dot-path config overrides, e.g. experiment=foo training.epochs=2")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, overrides=args.overrides)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    result = train(config, seed=seed, smoke=args.smoke, limit_batches=args.limit_batches,
                   device_override=args.device)
    print(json.dumps(to_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
