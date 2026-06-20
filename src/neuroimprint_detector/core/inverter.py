"""Gradient Inversion — Reconstruct training samples from NeuroImprint backdoor.

Implements the closed-form inversion from the NeuroImprint paper:
    x̃ = (W̃₂ - W₂) / (b̃₂ - b₂)

Where the difference between trained and original weights contains
the gradients of the memorized training samples.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InversionResult:
    """Result of gradient inversion."""
    embeddings: np.ndarray           # Reconstructed embedding vectors
    n_samples: int                   # Number of samples reconstructed
    inversion_quality: float         # Quality score (0-1)
    per_sample_quality: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class GradientInverter:
    """Inverts NeuroImprint backdoor to recover training sample embeddings.

    The NeuroImprint attack stores per-sample gradients in isolated neurons.
    After fine-tuning, the weight difference ΔW = W̃₂ - W₂ contains the
    gradients, which can be analytically inverted:

        x̃ = ΔW / Δb  (element-wise division)

    For SGD optimizer: exact reconstruction (raw gradients preserved)
    For Adam/AdamW: approximate reconstruction (sign of gradients only)
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def invert(
        self,
        W_original: np.ndarray,
        W_trained: np.ndarray,
        b_original: np.ndarray,
        b_trained: np.ndarray,
        optimizer: str = "sgd",
    ) -> InversionResult:
        """Perform closed-form gradient inversion.

        Args:
            W_original: Original W₂ weights (before fine-tuning)
            W_trained: Trained W₂ weights (after fine-tuning)
            b_original: Original b₂ biases
            b_trained: Trained b₂ biases
            optimizer: "sgd" (exact) or "adam"/"adamw" (approximate)

        Returns:
            InversionResult with reconstructed embeddings.
        """
        warnings = []

        # Validate shapes
        if W_original.shape != W_trained.shape:
            warnings.append(f"Shape mismatch: W_original {W_original.shape} vs W_trained {W_trained.shape}")
            return InversionResult(
                embeddings=np.array([]),
                n_samples=0,
                inversion_quality=0.0,
                warnings=warnings,
            )

        if b_original.shape != b_trained.shape:
            warnings.append(f"Shape mismatch: b_original {b_original.shape} vs b_trained {b_trained.shape}")
            return InversionResult(
                embeddings=np.array([]),
                n_samples=0,
                inversion_quality=0.0,
                warnings=warnings,
            )

        # Compute differences
        delta_W = W_trained - W_original  # ΔW = W̃₂ - W₂
        delta_b = b_trained - b_original  # Δb = b̃₂ - b₂

        # Check for zero differences (no backdoor or no training)
        if np.allclose(delta_W, 0, atol=self.epsilon):
            warnings.append("ΔW is zero — no gradients stored or no training occurred")
            return InversionResult(
                embeddings=np.array([]),
                n_samples=0,
                inversion_quality=0.0,
                warnings=warnings,
            )

        # Invert: x̃ = ΔW / Δb
        # Reshape delta_b for broadcasting: (n_bins,) -> (n_bins, 1)
        delta_b_reshaped = delta_b.reshape(-1, 1)

        # Avoid division by zero
        delta_b_safe = np.where(
            np.abs(delta_b_reshaped) < self.epsilon,
            self.epsilon,
            delta_b_reshaped,
        )

        # Element-wise division
        embeddings = delta_W / delta_b_safe

        # Compute quality metrics
        n_samples = embeddings.shape[0]
        per_sample_quality = []

        for i in range(n_samples):
            sample_emb = embeddings[i]
            # Quality: norm of the embedding (higher = more information)
            norm = np.linalg.norm(sample_emb)
            # Normalize to 0-1 range (assuming typical embedding norms)
            quality = min(1.0, norm / 10.0)
            per_sample_quality.append(float(quality))

        # Overall quality
        if per_sample_quality:
            inversion_quality = float(np.mean(per_sample_quality))
        else:
            inversion_quality = 0.0

        # Optimizer-specific quality adjustment
        if optimizer.lower() in ("adam", "adamw"):
            # Adam loses magnitude information (only sign preserved)
            inversion_quality *= 0.7
            warnings.append(
                "Adam/AdamW optimizer: reconstruction is approximate "
                "(magnitude information lost due to momentum)"
            )

        return InversionResult(
            embeddings=embeddings,
            n_samples=n_samples,
            inversion_quality=inversion_quality,
            per_sample_quality=per_sample_quality,
            warnings=warnings,
        )

    def embeddings_to_tokens(
        self,
        embeddings: np.ndarray,
        embedding_matrix: np.ndarray,
        method: str = "cosine",
    ) -> list[list[int]]:
        """Map reconstructed embeddings back to token sequences.

        Args:
            embeddings: Reconstructed embedding vectors (n_samples, hidden_dim)
            embedding_matrix: Model's embedding matrix (vocab_size, hidden_dim)
            method: "cosine" or "euclidean" for nearest neighbor search

        Returns:
            List of token ID sequences (one per sample).
        """
        if embeddings.size == 0 or embedding_matrix.size == 0:
            return []

        # Normalize for cosine similarity
        if method == "cosine":
            emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            emb_norms = np.where(emb_norms == 0, 1, emb_norms)
            embeddings_norm = embeddings / emb_norms

            vocab_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            vocab_norms = np.where(vocab_norms == 0, 1, vocab_norms)
            vocab_norm = embedding_matrix / vocab_norms

            # Cosine similarity: (n_samples, vocab_size)
            similarity = embeddings_norm @ vocab_norm.T

            # Nearest token for each embedding
            token_ids = np.argmax(similarity, axis=1).tolist()

            return [[tid] for tid in token_ids]

        else:  # euclidean
            token_ids = []
            for emb in embeddings:
                distances = np.linalg.norm(embedding_matrix - emb, axis=1)
                nearest = int(np.argmin(distances))
                token_ids.append([nearest])
            return token_ids
