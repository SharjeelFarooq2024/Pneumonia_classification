"""Algorithmic stratified sampling with quota discrepancy minimization."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np


class StratifiedSampler:
    """Discrepancy-minimizing stratified sampling algorithm.

    Applies the Largest Remainder Method (Hare-Niemeyer quota allocation)
    to allocate integer sample counts per class under strict budget fractions,
    guaranteeing that rounding errors do not disproportionately bias minority classes.
    """

    @staticmethod
    def compute_class_quotas(labels: np.ndarray, fraction: float) -> Dict[int, int]:
        """Compute exact integer counts per class minimizing total distribution discrepancy."""
        unique_classes, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        target_total = int(round(total_samples * fraction))

        # Raw non-integer quotas
        exact_quotas = {cls: (cnt / total_samples) * target_total for cls, cnt in zip(unique_classes, counts)}
        floor_quotas = {cls: int(math.floor(q)) for cls, q in exact_quotas.items()}
        allocated = sum(floor_quotas.values())
        remainder = target_total - allocated

        # Sort classes by largest fractional remainder: O(K log K) where K is number of classes
        remainders = [(exact_quotas[cls] - floor_quotas[cls], cls) for cls in unique_classes]
        remainders.sort(reverse=True, key=lambda x: x[0])

        # Distribute remaining slots to classes with largest fractional remainder
        final_quotas = dict(floor_quotas)
        for i in range(remainder):
            cls_to_increment = remainders[i][1]
            final_quotas[cls_to_increment] += 1

        return final_quotas

    @classmethod
    def sample_indices(cls, labels: np.ndarray, fraction: float, seed: int = 42) -> np.ndarray:
        """Select stratified indices preserving class proportions deterministically."""
        labels = np.asarray(labels, dtype=int)
        n = len(labels)
        if n == 0:
            raise ValueError("Cannot sample from an empty label array.")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"Fraction must be in (0, 1], got {fraction}")
        if fraction == 1.0:
            return np.arange(n, dtype=int)

        quotas = cls.compute_class_quotas(labels, fraction)
        rng = np.random.default_rng(seed)

        selected_indices: List[int] = []
        for class_id, quota in quotas.items():
            # Get all indices belonging to this class
            class_indices = np.where(labels == class_id)[0]
            # Permute deterministically and take exactly the quota
            chosen = rng.choice(class_indices, size=quota, replace=False)
            selected_indices.extend(chosen.tolist())

        # Sort final indices for reproducible deterministic memory access
        return np.sort(np.array(selected_indices, dtype=int))
