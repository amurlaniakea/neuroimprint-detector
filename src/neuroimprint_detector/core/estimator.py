"""Original weight estimation for NeuroImprint backdoor reconstruction.

In a real audit scenario, the auditor only has access to the trained adapter
(weights AFTER fine-tuning). The original backdoor weights (W2, b2 before
fine-tuning) are unknown.

This module estimates the original weights from the trained adapter using
the known structural properties of the NeuroImprint backdoor:
- W2 has identical row vectors (r2 repeated) before training
- b2 is sorted in ascending intervals before training
- After training: W2[i] = r2 + gradient_i for memorized samples
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class EstimatedOriginalWeights:
    """Estimated original weights of a NeuroImprint backdoor."""
    W2_original: np.ndarray
    b2_original: np.ndarray
    r2_estimate: np.ndarray          # The repeated row vector
    n_memorized_estimated: int       # Estimated number of memorized samples
    estimation_quality: float        # Confidence in the estimation (0-1)
    warnings: list[str]


def estimate_original_weights(
    W2_trained: Union[np.ndarray, torch.Tensor],
    b2_trained: Union[np.ndarray, torch.Tensor],
) -> EstimatedOriginalWeights:
    """Estimate the original (pre-training) weights of a NeuroImprint backdoor.

    The NeuroImprint attack creates W2 with identical row vectors (r2 repeated)
    and b2 sorted in ascending intervals. After training, memorized samples
    modify specific rows of W2 and elements of b2.

    Estimation strategy:
    1. r2_estimate = mean of all rows of W2 (non-memorized rows dominate)
    2. Identify memorized rows as those that deviate significantly from r2
    3. b2_original = b2_trained with gradient contributions removed

    Args:
        W2_trained: Trained W2 weight matrix (n_bins, hidden_dim)
        b2_trained: Trained b2 bias vector (n_bins,)

    Returns:
        EstimatedOriginalWeights with reconstructed original weights.
    """
    # Convert to numpy
    if isinstance(W2_trained, torch.Tensor):
        W2_trained = W2_trained.detach().cpu().numpy()
    if isinstance(b2_trained, torch.Tensor):
        b2_trained = b2_trained.detach().cpu().numpy()

    W2_trained = np.asarray(W2_trained, dtype=np.float64)
    b2_trained = np.asarray(b2_trained, dtype=np.float64)

    warnings = []

    if W2_trained.ndim != 2:
        warnings.append(f"W2 expected 2D, got {W2_trained.ndim}D")
        return EstimatedOriginalWeights(
            W2_original=W2_trained,
            b2_original=b2_trained,
            r2_estimate=np.zeros(W2_trained.shape[1] if W2_trained.ndim >= 2 else 0),
            n_memorized_estimated=0,
            estimation_quality=0.0,
            warnings=warnings,
        )

    n_bins, hidden_dim = W2_trained.shape
    if n_bins < 2:
        warnings.append("Too few bins for reliable estimation")
        return EstimatedOriginalWeights(
            W2_original=W2_trained,
            b2_original=b2_trained,
            r2_estimate=W2_trained[0],
            n_memorized_estimated=0,
            estimation_quality=0.1,
            warnings=warnings,
        )

    # Step 1: Estimate r2 as the median row (robust to outliers from memorization)
    r2_estimate = np.median(W2_trained, axis=0)

    # Step 2: Identify memorized rows
    # Compute deviation of each row from r2_estimate
    deviations = np.linalg.norm(W2_trained - r2_estimate, axis=1)

    # Use IQR-based outlier detection
    q1 = np.percentile(deviations, 25)
    q3 = np.percentile(deviations, 75)
    iqr = q3 - q1
    outlier_threshold = q3 + 3.0 * iqr  # Conservative threshold

    memorized_mask = deviations > outlier_threshold
    n_memorized = int(np.sum(memorized_mask))

    # Step 3: Reconstruct W2_original
    # Non-memorized rows should be identical to r2
    # For memorized rows, replace with r2_estimate (we lose the gradient info)
    W2_original = W2_trained.copy()
    W2_original[memorized_mask] = r2_estimate

    # Step 4: Reconstruct b2_original
    # The biases are sorted intervals. After training, memorized samples
    # shift specific bias elements. We estimate the original by re-sorting.
    b2_original = np.sort(b2_trained)

    # Quality estimation
    # High quality if: many non-memorized rows, clear separation
    non_memorized_ratio = 1.0 - (n_memorized / n_bins)
    if n_memorized > 0:
        # Check if memorized rows are clearly separated
        mem_dev = deviations[memorized_mask]
        non_mem_dev = deviations[~memorized_mask]
        separation = (np.mean(mem_dev) - np.mean(non_mem_dev)) / (np.std(non_mem_dev) + 1e-10)
        quality = min(1.0, non_memorized_ratio * 0.7 + min(separation / 5.0, 0.3))
    else:
        quality = 0.5  # No memorization detected, can't verify

    if n_memorized == 0:
        warnings.append("No memorized samples detected — adapter may be clean")
    elif non_memorized_ratio < 0.5:
        warnings.append(
            f"High memorization ratio ({n_memorized}/{n_bins}) — "
            "estimation quality may be degraded"
        )

    return EstimatedOriginalWeights(
        W2_original=W2_original.astype(np.float32),
        b2_original=b2_original.astype(np.float32),
        r2_estimate=r2_estimate.astype(np.float32),
        n_memorized_estimated=n_memorized,
        estimation_quality=float(quality),
        warnings=warnings,
    )


def compute_reconstruction_from_estimate(
    W2_trained: Union[np.ndarray, torch.Tensor],
    b2_trained: Union[np.ndarray, torch.Tensor],
    estimate: EstimatedOriginalWeights,
) -> np.ndarray:
    """Compute reconstructed embeddings using estimated original weights.

    Args:
        W2_trained: Trained W2 weights
        b2_trained: Trained b2 biases
        estimate: Estimated original weights

    Returns:
        Reconstructed embedding vectors (n_memorized, hidden_dim)
    """
    if isinstance(W2_trained, torch.Tensor):
        W2_trained = W2_trained.detach().cpu().numpy()
    if isinstance(b2_trained, torch.Tensor):
        b2_trained = b2_trained.detach().cpu().numpy()

    W2_trained = np.asarray(W2_trained, dtype=np.float64)
    b2_trained = np.asarray(b2_trained, dtype=np.float64)

    # delta_W = W2_trained - W2_original
    delta_W = W2_trained - estimate.W2_original

    # delta_b = b2_trained - b2_original
    delta_b = b2_trained - estimate.b2_original

    # Avoid division by zero
    delta_b_safe = np.where(np.abs(delta_b) < 1e-8, 1e-8, delta_b)

    # x = delta_W / delta_b (broadcasting: each row divided by corresponding delta_b)
    embeddings = delta_W / delta_b_safe[:, np.newaxis]

    return embeddings.astype(np.float32)
