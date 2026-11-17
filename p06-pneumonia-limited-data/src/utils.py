"""Shared utilities: config loading, seeding, logging, metrics, and evaluation."""
from __future__ import annotations

import copy
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def resolve_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def experiment_results_dir(config: Mapping[str, Any], experiment: str) -> Path:
    """Route experiments into the required results/ sub-structure."""
    base = resolve_path(Path(config.get("paths", {}).get("results", "results")))
    if experiment in ("baseline", "proposed"):
        return base / experiment
    if "_smoke" in experiment:
        return base / "smoke" / experiment
    if experiment.startswith("ablation_") or "_dataeff_" in experiment:
        return base / "ablations" / experiment
    return base / experiment


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path, seen: Optional[set] = None) -> Dict[str, Any]:
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular base_config chain involving {path}")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("base_config", None)
    if base_name:
        base_path = resolve_path(base_name)
        cfg = deep_merge(_load_yaml(base_path, seen), cfg)
    return cfg


def set_by_dotted_key(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _parse_override_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def load_config(path: str | os.PathLike[str], overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    cfg = _load_yaml(resolve_path(path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        set_by_dotted_key(cfg, key.strip(), _parse_override_value(value.strip()))
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility and environment
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def worker_init_fn(worker_id: int) -> None:
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def get_device(requested: str = "auto") -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_is_available(config: Mapping[str, Any], device: torch.device) -> bool:
    setting = config.get("training", {}).get("amp", "auto")
    if setting is True:
        return device.type == "cuda"
    if setting is False:
        return False
    return device.type == "cuda"


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    names = ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "matplotlib",
             "seaborn", "PyYAML", "tqdm", "medimeta", "Pillow"]
    for name in names:
        try:
            from importlib.metadata import version
            versions[name] = version(name)
        except Exception:
            versions[name] = "unknown"
    return versions


def environment_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "executable": sys.executable,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "versions": package_versions(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        props = torch.cuda.get_device_properties(0)
        info["cuda_total_memory_gb"] = round(props.total_memory / 1e9, 2)
    return info


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def dump_environment(config: Mapping[str, Any], device: torch.device, log_dir: str | os.PathLike[str] = "logs") -> Dict[str, Any]:
    out_dir = resolve_path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = environment_info(device)
    info["git_commit"] = git_commit()
    with open(out_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with open(out_dir / "environment.txt", "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {info['timestamp']}\n")
        f.write(f"Platform: {info['platform']}\n")
        f.write(f"Device: {info['device']}\n")
        f.write(f"CPU count: {info['cpu_count']}\n")
        f.write(f"Git commit: {info['git_commit']}\n")
        f.write("Versions:\n")
        for k, v in info["versions"].items():
            f.write(f"  {k}: {v}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA device: {info['cuda_device_name']}\n")
            f.write(f"CUDA version: {info['cuda_version']}\n")
            f.write(f"CUDA memory (GB): {info['cuda_total_memory_gb']}\n")
    return info


# ---------------------------------------------------------------------------
# Logging and timing
# ---------------------------------------------------------------------------
def setup_logger(name: str, log_file: Optional[str | os.PathLike[str]] = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        path = resolve_path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@contextmanager
def timer():
    start = time.time()
    yield lambda: time.time() - start


# ---------------------------------------------------------------------------
# Numeric helpers / metrics
# ---------------------------------------------------------------------------
def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | os.PathLike[str]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def expected_calibration_error(y_true: Sequence[int], y_prob: np.ndarray, n_bins: int = 15) -> Tuple[float, List[Dict[str, float]]]:
    y_true = np.asarray(y_true)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows: List[Dict[str, float]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(acc - avg_conf)
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": int(mask.sum()),
                     "accuracy": acc, "confidence": avg_conf})
    return float(ece), rows


def compute_metrics(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    class_names: Sequence[str],
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(class_names)))
    n_classes = len(class_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0, cm / cm.sum(axis=1, keepdims=True), 0.0)

    aurocs: Dict[str, float] = {}
    valid_aurocs: List[float] = []
    for k, name in enumerate(class_names):
        binary_true = (y_true == k).astype(int)
        if binary_true.min() == binary_true.max():
            aurocs[name] = float("nan")
            continue
        auc = float(roc_auc_score(binary_true, y_prob[:, k]))
        aurocs[name] = auc
        valid_aurocs.append(auc)
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")

    per_class: Dict[str, Dict[str, Any]] = {}
    for k, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[k]),
            "recall": float(recall[k]),
            "f1": float(f1[k]),
            "support": int(support[k]),
            "auroc_ovr": aurocs[name],
        }

    ece, reliability = expected_calibration_error(y_true, y_prob)

    def macro_f1_on(indices: np.ndarray) -> float:
        return float(f1_score(y_true[indices], y_pred[indices], labels=labels,
                              average="macro", zero_division=0))

    n = len(y_true)
    if bootstrap_n and bootstrap_n > 0:
        rng = np.random.default_rng(bootstrap_seed)
        boot = np.empty(bootstrap_n, dtype=float)
        for b in range(bootstrap_n):
            idx = rng.integers(0, n, size=n)
            boot[b] = macro_f1_on(idx)
        ci95 = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        boot_std = float(boot.std())
    else:
        ci95 = [float("nan"), float("nan")]
        boot_std = float("nan")

    metrics: Dict[str, Any] = {
        "n_samples": int(n),
        "class_names": list(class_names),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_auroc": macro_auroc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "confusion_matrix_labels": list(class_names),
        "ece": ece,
        "reliability_bins": reliability,
        "macro_f1_bootstrap_ci95": ci95,
        "macro_f1_bootstrap_std": boot_std,
        "support_per_class": {name: int(support[k]) for k, name in enumerate(class_names)},
    }
    return to_jsonable(metrics)


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, probabilities, image_indices) for a DataLoader."""
    model.eval()
    all_true: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []
    all_idx: List[np.ndarray] = []
    for batch in loader:
        if len(batch) == 3:
            images, targets, indices = batch
        else:  # pragma: no cover - defensive
            images, targets = batch
            indices = torch.arange(len(targets))
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_prob.append(probs)
        all_true.append(targets.numpy())
        all_idx.append(indices.numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_prob),
        np.concatenate(all_idx),
    )


def save_predictions_csv(
    path: str | os.PathLike[str],
    y_true: Sequence[int],
    y_prob: np.ndarray,
    image_indices: Sequence[int],
    class_names: Sequence[str],
) -> None:
    import pandas as pd

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_prob).argmax(axis=1)
    data: Dict[str, Any] = {
        "image_index": np.asarray(image_indices),
        "true_label": y_true,
        "predicted_label": y_pred,
        "true_class": [class_names[i] for i in y_true],
        "predicted_class": [class_names[i] for i in y_pred],
        "confidence": np.asarray(y_prob).max(axis=1),
    }
    for k, name in enumerate(class_names):
        data[f"prob_{name}"] = np.asarray(y_prob)[:, k]
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def aggregate_seed_metrics(seed_metrics: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    rows = list(seed_metrics)
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.array([float(r[key]) for r in rows if r.get(key) is not None and not np.isnan(float(r[key]))])
        if len(vals) == 0:
            out[key] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "n": int(len(vals))}
    return out


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def probe_batch_size(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> int:
    """Start from the configured batch size; fall back on CUDA OOM."""
    candidates = [int(config["training"].get("batch_size", 64))]
    candidates += [int(x) for x in config["training"].get("batch_size_fallbacks", [32, 16])]
    candidates = list(dict.fromkeys(candidates))
    if device.type != "cuda":
        logger.info("Device is CPU: using batch size %d (no VRAM-based fallback needed).", candidates[0])
        return candidates[0]

    image_size = int(config["data"].get("image_size", 224))
    channels = int(config["model"].get("input_channels", 3))
    for batch_size in candidates:
        try:
            model.zero_grad(set_to_none=True)
            dummy = torch.randn(batch_size, channels, image_size, image_size, device=device)
            with torch.autocast(device_type="cuda", enabled=amp_is_available(config, device)):
                out = model(dummy)
                loss = out.float().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            del dummy, out, loss
            torch.cuda.empty_cache()
            logger.info("Selected batch size %d based on a CUDA memory probe.", batch_size)
            return batch_size
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning("CUDA OOM with batch size %d; trying a smaller batch.", batch_size)
            torch.cuda.empty_cache()
    raise RuntimeError("Out of CUDA memory for every configured batch size fallback.")
