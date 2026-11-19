"""Probability calibration algorithms utilizing efficient binning data structures."""
from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np

from src.domain.entities import CalibrationBin
from src.domain.interfaces import ICalibrationAlgorithm


class FastHistogramCalibration(ICalibrationAlgorithm):
    """Optimal O(N + B) single-pass histogram calibration algorithm.

    Replaces naive repeated boolean masking with a direct bucket index calculation
    and array-based accumulation, avoiding O(B * N) passes over the dataset.
    """

    def compute_ece(
        self,
        y_true: Sequence[int],
        y_prob: np.ndarray,
        n_bins: int = 15,
    ) -> Tuple[float, List[CalibrationBin]]:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        n = len(y_true)
        if n == 0:
            return 0.0, []

        conf = y_prob.max(axis=1)
        pred = y_prob.argmax(axis=1)
        correct = (pred == y_true).astype(float)

        # Allocate O(B) bucket accumulators
        counts = np.zeros(n_bins, dtype=int)
        conf_sums = np.zeros(n_bins, dtype=float)
        acc_sums = np.zeros(n_bins, dtype=float)

        # Compute bucket indices in O(N): bin = floor(conf * n_bins), clamped to [0, n_bins - 1]
        bin_indices = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)

        # Vectorized accumulation using np.add.at
        np.add.at(counts, bin_indices, 1)
        np.add.at(conf_sums, bin_indices, conf)
        np.add.at(acc_sums, bin_indices, correct)

        bin_width = 1.0 / n_bins
        bins: List[CalibrationBin] = []
        total_ece = 0.0

        for b in range(n_bins):
            cnt = int(counts[b])
            if cnt == 0:
                continue
            mean_c = float(conf_sums[b] / cnt)
            acc = float(acc_sums[b] / cnt)
            weight = float(cnt / n)
            total_ece += weight * abs(acc - mean_c)

            bins.append(
                CalibrationBin(
                    bin_index=b,
                    bin_lower=round(b * bin_width, 4),
                    bin_upper=round((b + 1) * bin_width, 4),
                    sample_count=cnt,
                    mean_confidence=mean_c,
                    accuracy=acc,
                )
            )

        return float(total_ece), bins


class AdaptiveQuantileCalibration(ICalibrationAlgorithm):
    """Equal-frequency (quantile) calibration algorithm.

    Sorts confidence values in O(N log N) and assigns equal numbers of samples
    to each bin, preventing empty bins in sparse confidence regions.
    """

    def compute_ece(
        self,
        y_true: Sequence[int],
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[float, List[CalibrationBin]]:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        n = len(y_true)
        if n == 0:
            return 0.0, []

        conf = y_prob.max(axis=1)
        pred = y_prob.argmax(axis=1)
        correct = (pred == y_true).astype(float)

        # Sort indices by confidence: O(N log N)
        sort_order = np.argsort(conf)
        sorted_conf = conf[sort_order]
        sorted_correct = correct[sort_order]

        bin_size = n / n_bins
        bins: List[CalibrationBin] = []
        total_ece = 0.0

        for b in range(n_bins):
            start_idx = int(round(b * bin_size))
            end_idx = int(round((b + 1) * bin_size))
            if start_idx >= end_idx:
                continue

            bin_conf = sorted_conf[start_idx:end_idx]
            bin_corr = sorted_correct[start_idx:end_idx]
            cnt = len(bin_conf)

            mean_c = float(bin_conf.mean())
            acc = float(bin_corr.mean())
            weight = float(cnt / n)
            total_ece += weight * abs(acc - mean_c)

            bins.append(
                CalibrationBin(
                    bin_index=b,
                    bin_lower=float(bin_conf.min()),
                    bin_upper=float(bin_conf.max()),
                    sample_count=cnt,
                    mean_confidence=mean_c,
                    accuracy=acc,
                )
            )

        return float(total_ece), bins
