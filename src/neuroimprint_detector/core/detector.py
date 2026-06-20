"""NeuroImprint Detector — Audit PEFT adapters for privacy backdoors.

Detects if a PEFT adapter contains a NeuroImprint backdoor that memorizes
training samples from federated learning clients.

Based on: Shi et al. (2026) "From Efficiency to Leakage — Privacy Backdoor
in Federated Language Model Fine-Tuning" — arXiv:2606.20553
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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
    has_bin_structure: bool = False          # W₂ with identical row vectors
    has_sorted_biases: bool = False          # Biases organized in intervals
    has_ralu_pattern: bool = False           # RaLU activation fingerprint
    estimated_samples: int = 0               # Estimated memorized samples
    reconstruction_possible: bool = False    # Can we reconstruct?
    # Metrics
    weight_symmetry_score: float = 0.0       # How symmetric are the weights
    bias_interval_score: float = 0.0         # How interval-like are the biases
    rality_score: float = 0.0                # Overall backdoor likelihood
    # Details
    details: dict = field(default_factory=dict)


class NeuroImprintDetector:
    """Detects NeuroImprint backdoors in PEFT adapters.

    The NeuroImprint attack creates a privacy backdoor in a parallel PEFT
    adapter by organizing neurons into "reconstruction bins" where each
    bin memorizes exactly one training sample.

    Detection strategy:
    1. Analyze weight matrix W₂ for identical row vectors (signature of backdoor)
    2. Analyze bias vector b₂ for sorted interval structure
    3. Check for RaLU activation pattern fingerprint
    4. Estimate number of memorized samples from bin structure
    """

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
        adapter_weights: dict[str, np.ndarray],
        adapter_config: Optional[dict] = None,
    ) -> DetectionResult:
        """Analyze an adapter for NeuroImprint backdoor.

        Args:
            adapter_weights: Dict mapping layer names to weight arrays.
                Expected keys: 'W1', 'W2', 'W3', 'b1', 'b2', 'b3'
                (or equivalent LoRA format: 'lora_A', 'lora_B')
            adapter_config: Optional adapter configuration (rank, alpha, etc.)

        Returns:
            DetectionResult with verdict and detailed analysis.
        """
        result = DetectionResult(
            verdict=Verdict.CLEAN,
            confidence=0.0,
            adapter_type=AdapterType.UNKNOWN,
        )

        # Identify adapter type
        result.adapter_type = self._identify_adapter_type(adapter_weights)

        # Signal 1: Weight bin structure (W₂ with identical rows)
        W2 = adapter_weights.get('W2', adapter_weights.get('lora_B', None))
        if W2 is not None:
            result.weight_symmetry_score = self._check_bin_structure(W2)
            result.has_bin_structure = result.weight_symmetry_score > self.bin_structure_threshold

        # Signal 2: Bias interval structure (b₂ sorted in intervals)
        b2 = adapter_weights.get('b2', adapter_weights.get('lora_bias', None))
        if b2 is not None:
            result.bias_interval_score = self._check_bias_intervals(b2)
            result.has_sorted_biases = result.bias_interval_score > self.bias_interval_threshold

        # Signal 3: RaLU activation pattern
        if W2 is not None and b2 is not None:
            result.has_ralu_pattern = self._check_ralu_pattern(W2, b2)

        # Combine signals
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

        # Estimate memorized samples
        if result.verdict in (Verdict.BACKDOORED, Verdict.SUSPICIOUS):
            result.estimated_samples = self._estimate_samples(W2, b2)
            result.reconstruction_possible = result.has_bin_structure and result.has_sorted_biases

        # Build details
        result.details = {
            "n_signals_detected": n_signals,
            "weight_shape": W2.shape if W2 is not None else None,
            "bias_shape": b2.shape if b2 is not None else None,
            "adapter_config": adapter_config,
        }

        return result

    def _identify_adapter_type(self, weights: dict[str, np.ndarray]) -> AdapterType:
        """Identify the type of PEFT adapter from weight names."""
        keys = set(weights.keys())
        if 'lora_A' in keys or 'lora_B' in keys:
            return AdapterType.LORA
        if 'W1' in keys and 'W2' in keys and 'W3' in keys:
            return AdapterType.PARALLEL
        if 'adapter_down' in keys or 'adapter_up' in keys:
            return AdapterType.SERIAL
        return AdapterType.UNKNOWN

    def _check_bin_structure(self, W: np.ndarray) -> float:
        """Check if weight matrix has the NeuroImprint bin structure.

        The NeuroImprint backdoor creates W₂ where all row vectors are
        identical (r₂ repeated). This is a signature of the attack.

        Returns score 0.0-1.0 indicating how likely this structure is present.
        """
        if W.ndim != 2:
            return 0.0

        n_rows, n_cols = W.shape
        if n_rows < 2:
            return 0.0

        # Compute pairwise cosine similarity between rows
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        W_normalized = W / norms

        # Cosine similarity matrix
        sim_matrix = W_normalized @ W_normalized.T

        # For identical rows, all off-diagonal elements should be ~1
        # Extract upper triangle (excluding diagonal)
        upper_tri = sim_matrix[np.triu_indices(n_rows, k=1)]

        if len(upper_tri) == 0:
            return 0.0

        # Score: mean similarity (1.0 = all rows identical)
        mean_sim = float(np.mean(upper_tri))
        # Clamp to [0, 1] (cosine similarity can be -1 to 1)
        score = max(0.0, min(1.0, (mean_sim + 1) / 2))

        return score

    def _check_bias_intervals(self, b: np.ndarray) -> float:
        """Check if bias vector has the interval structure of NeuroImprint.

        The NeuroImprint backdoor organizes biases in sorted intervals:
        b₂ = -[F⁻¹(1/m), F⁻¹(2/m), ..., F⁻¹(1)]ᵀ

        This means b₂ should be sorted in ascending order with roughly
        equal spacing (quantile-based).

        Returns score 0.0-1.0.
        """
        if b.ndim != 1 or len(b) < 2:
            return 0.0

        b_flat = b.flatten()

        # Check if sorted (ascending)
        sorted_b = np.sort(b_flat)
        is_sorted = np.allclose(b_flat, sorted_b, atol=1e-6) or \
                    np.allclose(b_flat, sorted_b[::-1], atol=1e-6)

        # Check interval regularity
        if len(b_flat) > 2:
            diffs = np.diff(sorted_b)
            # For equal intervals, all diffs should be similar
            mean_diff = np.mean(diffs)
            if mean_diff > 0:
                cv = np.std(diffs) / mean_diff  # coefficient of variation
                regularity_score = max(0.0, 1.0 - cv)
            else:
                regularity_score = 0.0
        else:
            regularity_score = 0.5

        # Combine sorted check with regularity
        sorted_score = 1.0 if is_sorted else 0.0
        score = 0.6 * sorted_score + 0.4 * regularity_score

        return float(score)

    def _check_ralu_pattern(self, W: np.ndarray, b: np.ndarray) -> bool:
        """Check for RaLU activation function fingerprint.

        RaLU (Ranged Linear Unit) limits activation to a specific range:
        RaLU(z) = z if 0 < z < upper_bound, else 0

        This creates a distinctive pattern where neurons are either
        inactive (zero) or active within a narrow range.

        Returns True if RaLU pattern is detected.
        """
        if W.ndim != 2 or b.ndim != 1:
            return False

        n_bins = min(W.shape[0], b.shape[0])
        if n_bins < 2:
            return False

        # Check if weights show the reduced-rank signature
        # The backdoor W₂ has rank 1 (all rows are scaled versions of r₂)
        try:
            U, S, Vt = np.linalg.svd(W[:n_bins], full_matrices=False)
            # Rank-1 matrix has one dominant singular value
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
        """Estimate the number of memorized training samples.

        In NeuroImprint, the number of reconstruction bins m determines
        how many samples can be memorized. Each bin corresponds to one sample.

        m is the dimension of the bias vector b₂ (or the number of rows in W₂).
        """
        if b is not None and b.ndim == 1:
            return int(b.shape[0])
        if W is not None and W.ndim == 2:
            return int(W.shape[0])
        return 0
