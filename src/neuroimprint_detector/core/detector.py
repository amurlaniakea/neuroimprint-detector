"""NeuroImprint Detector — Audit PEFT adapters for privacy backdoors.

Detects if a PEFT adapter contains a NeuroImprint backdoor that memorizes
training samples from federated learning clients.

Based on: Shi et al. (2026) "From Efficiency to Leakage — Privacy Backdoor
in Federated Language Model Fine-Tuning" — arXiv:2606.20553
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import numpy as np


class AdapterType(Enum):
    LORA = "lora"
    PARALLEL = "parallel"
    SERIAL = "serial"
    UNKNOWN = "unknown"


class Verdict(Enum):
    CLEAN = "clean"
    BACKDOORED = "backdoored"
    SUSPICIOUS = "suspicious"


@dataclass
class DetectionResult:
    """Result of analyzing an adapter for NeuroImprint backdoor."""
    verdict: Verdict
    confidence: float  # 0.0 to 1.0
    adapter_type: AdapterType
    # Detection signals
    has_bin_structure: bool = False
    has_sorted_biases: bool = False
    has_ralu_pattern: bool = False
    estimated_samples: int = 0
    reconstruction_possible: bool = False
    # Metrics
    weight_symmetry_score: float = 0.0
    bias_interval_score: float = 0.0
    rality_score: float = 0.0
    # Details
    details: dict = field(default_factory=dict)


def _to_numpy(tensor: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Convert a tensor (numpy or torch) to numpy array."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


class NeuroImprintDetector:
    """Detects NeuroImprint backdoors in PEFT adapters."""

    def __init__(
        self,
        bin_structure_threshold: float = 0.85,
        bias_interval_threshold: float = 0.80,
        min_confidence: float = 0.5,
    ):
        self.bin_structure_threshold = bin_structure_threshold
        self.bias_interval_threshold = bias_interval_threshold
        self.min_confidence = min_confidence

    def analyze(
        self,
        adapter_weights: dict[str, Union[np.ndarray, "torch.Tensor"]],
        adapter_config: Optional[dict] = None,
    ) -> DetectionResult:
        """Analyze an adapter for NeuroImprint backdoor."""
        result = DetectionResult(
            verdict=Verdict.CLEAN,
            confidence=0.0,
            adapter_type=AdapterType.UNKNOWN,
        )

        result.adapter_type = self._identify_adapter_type(adapter_weights)

        W2 = adapter_weights.get('W2', adapter_weights.get('lora_B', None))
        if W2 is not None:
            W2_np = _to_numpy(W2)
            result.weight_symmetry_score = self._check_bin_structure(W2_np)
            result.has_bin_structure = result.weight_symmetry_score > self.bin_structure_threshold

        b2 = adapter_weights.get('b2', adapter_weights.get('lora_bias', None))
        if b2 is not None:
            b2_np = _to_numpy(b2)
            result.bias_interval_score = self._check_bias_intervals(b2_np)
            result.has_sorted_biases = result.bias_interval_score > self.bias_interval_threshold

        if W2 is not None and b2 is not None:
            result.has_ralu_pattern = self._check_ralu_pattern(_to_numpy(W2), _to_numpy(b2))

        signals = [
            result.has_bin_structure,
            result.has_sorted_biases,
            result.has_ralu_pattern,
        ]
        n_signals = sum(signals)

        if n_signals >= 2:
            result.verdict = Verdict.BACKDOORED
            result.confidence = min(0.5 + 0.2 * n_signals, 0.95)
        elif n_signals == 1:
            result.verdict = Verdict.SUSPICIOUS
            result.confidence = 0.5
        else:
            result.verdict = Verdict.CLEAN
            result.confidence = 0.9

        if result.verdict in (Verdict.BACKDOORED, Verdict.SUSPICIOUS):
            result.estimated_samples = self._estimate_samples(
                _to_numpy(W2) if W2 is not None else None,
                _to_numpy(b2) if b2 is not None else None,
            )
            result.reconstruction_possible = result.has_bin_structure and result.has_sorted_biases

        result.details = {
            "n_signals_detected": n_signals,
            "weight_shape": _to_numpy(W2).shape if W2 is not None else None,
            "bias_shape": _to_numpy(b2).shape if b2 is not None else None,
            "adapter_config": adapter_config,
        }

        return result

    def _identify_adapter_type(self, weights: dict) -> AdapterType:
        keys = set(weights.keys())
        if 'lora_A' in keys or 'lora_B' in keys:
            return AdapterType.LORA
        if 'W1' in keys and 'W2' in keys and 'W3' in keys:
            return AdapterType.PARALLEL
        if 'adapter_down' in keys or 'adapter_up' in keys:
            return AdapterType.SERIAL
        return AdapterType.UNKNOWN

    def _check_bin_structure(self, W: np.ndarray) -> float:
        if W.ndim != 2 or W.shape[0] < 2:
            return 0.0
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        W_normalized = W / norms
        sim_matrix = W_normalized @ W_normalized.T
        upper_tri = sim_matrix[np.triu_indices(W.shape[0], k=1)]
        if len(upper_tri) == 0:
            return 0.0
        mean_sim = float(np.mean(upper_tri))
        return max(0.0, min(1.0, (mean_sim + 1) / 2))

    def _check_bias_intervals(self, b: np.ndarray) -> float:
        if b.ndim != 1 or len(b) < 2:
            return 0.0
        b_flat = b.flatten()
        sorted_b = np.sort(b_flat)
        is_sorted = np.allclose(b_flat, sorted_b, atol=1e-6) or \
                    np.allclose(b_flat, sorted_b[::-1], atol=1e-6)
        if len(b_flat) > 2:
            diffs = np.diff(sorted_b)
            mean_diff = np.mean(diffs)
            if mean_diff > 0:
                cv = np.std(diffs) / mean_diff
                regularity_score = max(0.0, 1.0 - cv)
            else:
                regularity_score = 0.0
        else:
            regularity_score = 0.5
        sorted_score = 1.0 if is_sorted else 0.0
        return float(0.6 * sorted_score + 0.4 * regularity_score)

    def _check_ralu_pattern(self, W: np.ndarray, b: np.ndarray) -> bool:
        if W.ndim != 2 or b.ndim != 1:
            return False
        n_bins = min(W.shape[0], b.shape[0])
        if n_bins < 2:
            return False
        try:
            U, S, Vt = np.linalg.svd(W[:n_bins], full_matrices=False)
            if S[0] > 0:
                dominance = S[0] / (np.sum(S) + 1e-10)
                return dominance > 0.8
        except Exception:
            pass
        return False

    def _estimate_samples(
        self,
        W: Optional[np.ndarray],
        b: Optional[np.ndarray],
    ) -> int:
        if b is not None and b.ndim == 1:
            return int(b.shape[0])
        if W is not None and W.ndim == 2:
            return int(W.shape[0])
        return 0
