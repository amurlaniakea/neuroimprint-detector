"""Synthetic data generator for NeuroImprint testing.

Generates clean and backdoored adapters with known ground truth for
calibrating detection thresholds and running integration tests.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SyntheticAdapter:
    """A synthetic adapter with known properties."""
    weights: dict[str, torch.Tensor]
    is_backdoored: bool
    n_memorized_samples: int = 0
    metadata: dict = field(default_factory=dict)


class SyntheticAdapterGenerator:
    """Generates synthetic adapters for testing NeuroImprint detection.

    Can create:
    - Clean adapters (random weights, no backdoor structure)
    - Backdoored adapters (with NeuroImprint L1→L2→L3 structure)
    - LoRA-disguised backdoors (backdoor hidden as LoRA weights)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate_clean(
        self,
        hidden_dim: int = 768,
        reduced_dim: int = 64,
        n_bins: int = 200,
    ) -> SyntheticAdapter:
        """Generate a clean adapter with random weights (no backdoor)."""
        weights = {
            'l1.weight': torch.randn(reduced_dim, hidden_dim),
            'l2.weight': torch.randn(n_bins, reduced_dim),
            'l2.bias': torch.randn(n_bins),
            'l3.weight': torch.randn(hidden_dim, n_bins),
        }

        return SyntheticAdapter(
            weights=weights,
            is_backdoored=False,
            n_memorized_samples=0,
            metadata={"type": "clean", "hidden_dim": hidden_dim, "n_bins": n_bins},
        )

    def generate_backdoored(
        self,
        hidden_dim: int = 768,
        reduced_dim: int = 64,
        n_bins: int = 200,
        n_memorized_samples: int = 0,
        optimizer: str = "sgd",
    ) -> SyntheticAdapter:
        """Generate an adapter with NeuroImprint backdoor structure.

        Args:
            hidden_dim: Model hidden dimension (h)
            reduced_dim: Reduced dimension (ĥ)
            n_bins: Number of reconstruction bins (m)
            n_memorized_samples: Number of samples to simulate as memorized
            optimizer: "sgd" or "adam" (affects gradient storage pattern)

        Returns:
            SyntheticAdapter with backdoor structure.
        """
        # L1: Projection layer (h → ĥ)
        # Use random projection (SVD-based like the paper)
        random_matrix = self.rng.randn(hidden_dim, reduced_dim)
        U, _, Vt = np.linalg.svd(random_matrix, full_matrices=False)
        W1 = torch.tensor(U[:, :reduced_dim] @ Vt[:reduced_dim, :], dtype=torch.float32)

        # L2: Memorization layer (ĥ → m)
        # Key signature: all row vectors are identical (r₂ repeated)
        r2 = torch.randn(reduced_dim)
        W2 = r2.unsqueeze(0).repeat(n_bins, 1)  # (n_bins, reduced_dim)

        # Biases: sorted intervals (quantile-based)
        # b₂ = -[F⁻¹(1/m), F⁻¹(2/m), ..., F⁻¹(1)]
        quantiles = np.linspace(1.0 / n_bins, 1.0, n_bins)
        b2 = -torch.tensor(quantiles, dtype=torch.float32)

        # L3: Output layer (m → h)
        # All row vectors identical, output canceled by LayerNorm
        r3 = torch.randn(hidden_dim)
        W3 = r3.unsqueeze(0).repeat(n_bins, 1)  # (n_bins, hidden_dim)

        weights = {
            'l1.weight': W1,
            'l2.weight': W2,
            'l2.bias': b2,
            'l3.weight': W3,
        }

        # Simulate memorized samples by adding gradients to L2
        if n_memorized_samples > 0:
            actual_samples = min(n_memorized_samples, n_bins)
            for i in range(actual_samples):
                # Each sample adds a gradient to its assigned neuron
                gradient = torch.randn(reduced_dim) * 0.01
                if optimizer == "sgd":
                    # SGD: exact gradient storage
                    weights['l2.weight'][i] += gradient
                else:
                    # Adam: sign-only storage (approximate)
                    weights['l2.weight'][i] += torch.sign(gradient) * 0.01

        return SyntheticAdapter(
            weights=weights,
            is_backdoored=True,
            n_memorized_samples=n_memorized_samples,
            metadata={
                "type": "backdoored",
                "hidden_dim": hidden_dim,
                "reduced_dim": reduced_dim,
                "n_bins": n_bins,
                "optimizer": optimizer,
            },
        )

    def generate_lora_disguised(
        self,
        hidden_dim: int = 768,
        rank: int = 64,
        n_bins: int = 200,
    ) -> SyntheticAdapter:
        """Generate a backdoor disguised as a LoRA adapter.

        The backdoor structure is hidden in lora_A and lora_B such that
        lora_B @ lora_A produces the NeuroImprint weight pattern.
        """
        # Create W2 with backdoor structure
        r2 = torch.randn(rank)
        W2_backdoor = r2.unsqueeze(0).repeat(n_bins, 1)

        # Decompose W2 into LoRA factors: W2 = lora_B @ lora_A
        # lora_A: (rank, reduced_dim), lora_B: (n_bins, rank)
        lora_A = torch.randn(rank, rank)
        lora_B = W2_backdoor @ torch.linalg.pinv(lora_A)

        # Create random L1 and L3
        lora_A_l1 = torch.randn(rank, hidden_dim)
        lora_B_l1 = torch.randn(rank, rank) @ lora_A_l1

        lora_A_l3 = torch.randn(hidden_dim, rank)
        lora_B_l3 = torch.randn(rank, rank)

        quantiles = np.linspace(1.0 / n_bins, 1.0, n_bins)
        b2 = -torch.tensor(quantiles, dtype=torch.float32)

        weights = {
            'lora_A': lora_A,
            'lora_B': lora_B,
            'lora_A_l1': lora_A_l1,
            'lora_B_l1': lora_B_l1,
            'lora_A_l3': lora_A_l3,
            'lora_B_l3': lora_B_l3,
            'l2.bias': b2,
        }

        return SyntheticAdapter(
            weights=weights,
            is_backdoored=True,
            n_memorized_samples=0,
            metadata={
                "type": "lora_disguised",
                "hidden_dim": hidden_dim,
                "rank": rank,
                "n_bins": n_bins,
            },
        )

    def generate_mixed_state_dict(
        self,
        hidden_dim: int = 768,
        reduced_dim: int = 64,
        n_bins: int = 200,
        n_memorized: int = 50,
    ) -> dict[str, torch.Tensor]:
        """Generate a realistic mixed state dict with both clean and backdoored layers.

        Simulates a model where the attacker injected a backdoor into
        specific transformer layers while leaving others clean.
        """
        state_dict = {}

        # Clean layers (random weights)
        for layer_idx in range(3):
            prefix = f"model.layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
            state_dict[f"{prefix}.k_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
            state_dict[f"{prefix}.v_proj.weight"] = torch.randn(hidden_dim, hidden_dim)

        # Backdoored layer (layer 1)
        backdoor_prefix = "model.layers.1.self_attn.privacy_backdoor"
        r2 = torch.randn(reduced_dim)
        W2 = r2.unsqueeze(0).repeat(n_bins, 1)
        quantiles = np.linspace(1.0 / n_bins, 1.0, n_bins)
        b2 = -torch.tensor(quantiles, dtype=torch.float32)

        state_dict[f"{backdoor_prefix}.l1.weight"] = torch.randn(reduced_dim, hidden_dim)
        state_dict[f"{backdoor_prefix}.l2.weight"] = W2
        state_dict[f"{backdoor_prefix}.l2.bias"] = b2
        state_dict[f"{backdoor_prefix}.l3.weight"] = torch.randn(hidden_dim, n_bins)

        # Add memorized samples
        for i in range(min(n_memorized, n_bins)):
            gradient = torch.randn(reduced_dim) * 0.01
            state_dict[f"{backdoor_prefix}.l2.weight"][i] += gradient

        return state_dict
