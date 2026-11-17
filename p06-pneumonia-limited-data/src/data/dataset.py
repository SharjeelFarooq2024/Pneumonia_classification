"""MedIMeta pediatric pneumonia data access and transforms."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils import resolve_path, worker_init_fn


def _identity(image):
    return image


def _build_base(config: Mapping[str, Any], split: str, transform=None):
    from medimeta import MedIMeta

    data_root = str(resolve_path(config["data"]["root"]))
    # MedIMeta replaces transform=None with its own default (which includes
    # ToTensor), so use an explicit no-op to keep PIL images for our pipeline.
    return MedIMeta(
        data_path=data_root,
        dataset_id=config["data"].get("dataset_id", "pneumonia"),
        task_name=config["data"].get("task_name", "disease class"),
        split=split,
        transform=transform if transform is not None else _identity,
        use_hdf5=False,
    )


def get_split_labels(config: Mapping[str, Any], split: str) -> np.ndarray:
    base = _build_base(config, split)
    return np.asarray(base.labels, dtype=int)


def get_split_global_indices(config: Mapping[str, Any], split: str) -> List[int]:
    base = _build_base(config, split)
    return [int(i) for i in base.split_indices]


class PneumoniaDataset(Dataset):
    """Thin wrapper around medimeta.MedIMeta that also returns the global image index.

    Grayscale TIFFs are replicated to three channels so that the baseline and the
    pretrained models share the same input shape.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        split: str,
        positions: Optional[Sequence[int]] = None,
        transform=None,
        use_hdf5: bool = False,
    ) -> None:
        self.config = dict(config)
        self.split = split
        self.transform = transform
        self.base = _build_base(config, split, transform=None)
        self.split_indices = [int(i) for i in self.base.split_indices]
        self.split_labels = np.asarray(self.base.labels, dtype=int)
        if positions is None:
            positions = list(range(len(self.split_labels)))
        self.positions = np.asarray(list(positions), dtype=int)
        if len(self.positions) and (self.positions.min() < 0 or self.positions.max() >= len(self.split_labels)):
            raise IndexError("Subset positions fall outside the requested split.")
        self.global_indices = [self.split_indices[p] for p in self.positions]
        self.labels = self.split_labels[self.positions] if len(self.positions) else np.array([], dtype=int)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int):
        position = int(self.positions[index])
        image, label = self.base[position]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label), int(self.global_indices[index])


def build_transform(config: Mapping[str, Any], train: bool = False) -> transforms.Compose:
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    image_size = int(data_cfg.get("image_size", 224))
    use_imagenet = bool(config.get("model", {}).get("imagenet_normalization", False))
    mean = [0.485, 0.456, 0.406] if use_imagenet else [0.5, 0.5, 0.5]
    std = [0.229, 0.224, 0.225] if use_imagenet else [0.5, 0.5, 0.5]

    steps: List[Any] = []
    do_augment = bool(train and train_cfg.get("augment", False))
    aug = train_cfg.get("augment_params", {}) or {}
    if do_augment:
        if aug.get("random_resized_crop", False):
            steps.append(transforms.RandomResizedCrop(
                image_size,
                scale=(float(aug.get("rrc_scale_min", 0.85)), float(aug.get("rrc_scale_max", 1.0))),
            ))
        else:
            steps.append(transforms.Resize((image_size, image_size)))
        if float(aug.get("rotation_deg", 0)) > 0:
            steps.append(transforms.RandomRotation(float(aug["rotation_deg"])))
        translate = float(aug.get("translate", 0.0))
        scale_min = float(aug.get("scale_min", 1.0))
        scale_max = float(aug.get("scale_max", 1.0))
        if translate > 0 or scale_min != 1.0 or scale_max != 1.0:
            steps.append(transforms.RandomAffine(
                degrees=0, translate=(translate, translate), scale=(scale_min, scale_max)
            ))
        if aug.get("horizontal_flip", False):
            steps.append(transforms.RandomHorizontalFlip(p=0.5))
        brightness = float(aug.get("brightness", 0.0))
        contrast = float(aug.get("contrast", 0.0))
        if brightness > 0 or contrast > 0:
            steps.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
    else:
        steps.append(transforms.Resize((image_size, image_size)))

    steps.extend([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(steps)


def make_dataloader(
    dataset: Dataset,
    config: Mapping[str, Any],
    train: bool,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    data_cfg = config.get("data", {})
    num_workers = int(data_cfg.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(int(seed if seed is not None else config.get("seed", 42)))
    return DataLoader(
        dataset,
        batch_size=int(batch_size or config["training"].get("batch_size", 64)),
        shuffle=bool(train),
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def make_datasets(
    config: Mapping[str, Any],
    split: str,
    positions: Optional[Sequence[int]] = None,
    train: bool = False,
) -> PneumoniaDataset:
    return PneumoniaDataset(
        config=config,
        split=split,
        positions=positions,
        transform=build_transform(config, train=train),
    )


def load_subset_positions(path: str | os.PathLike[str]) -> List[int]:
    import json

    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Subset file not found: {path}. Run `python scripts/make_subset.py` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "positions" in payload:
        return [int(x) for x in payload["positions"]]
    # Fallback: treat the stored global indices as positions only if they are contiguous.
    raise KeyError("Subset JSON must contain a 'positions' field (positions within the split).")
