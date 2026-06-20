"""Hugging Face / PEFT adapter loader for NeuroImprint detection.

Extracts weight tensors from PEFT adapters (LoRA, parallel, serial) and
identifies potential NeuroImprint backdoor structures.

Supports:
- Loading from .safetensors / .bin files on disk
- Loading from in-memory Hugging Face models (with or without PEFT)
- Automatic detection of backdoor layer patterns in state dicts
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ExtractedLayer:
    """A extracted layer from an adapter."""
    name: str
    weight: torch.Tensor
    bias: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BackdoorCandidate:
    """A potential NeuroImprint backdoor structure found in an adapter."""
    prefix: str
    W1: Optional[torch.Tensor] = None
    W2: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    W3: Optional[torch.Tensor] = None
    lora_A: Optional[torch.Tensor] = None  # If backdoor is disguised as LoRA
    lora_B: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)

    @property
    def has_full_structure(self) -> bool:
        """Check if all 3 layers (L1, L2, L3) are present."""
        return self.W1 is not None and self.W2 is not None and self.W3 is not None

    @property
    def has_memorization_layer(self) -> bool:
        """Check if the memorization layer (L2) is present."""
        return self.W2 is not None and self.b2 is not None

    @property
    def equivalent_W2(self) -> Optional[torch.Tensor]:
        """Get the equivalent W2 matrix (handles LoRA disguise)."""
        if self.W2 is not None:
            return self.W2
        if self.lora_A is not None and self.lora_B is not None:
            return self.lora_B @ self.lora_A
        return None


class AdapterExtractor:
    """Extracts and analyzes PEFT adapter weights for NeuroImprint backdoor.

    Works with raw state_dict tensors, supporting both disk-loaded files
    and in-memory Hugging Face models.
    """

    # Known naming patterns for adapter layers
    LORA_PATTERNS = [
        r'(.+)\.lora_A\.(.+)\.weight$',
        r'(.+)\.lora_B\.(.+)\.weight$',
        r'(.+)\.lora_A\.weight$',
        r'(.+)\.lora_B\.weight$',
    ]

    PARALLEL_PATTERNS = [
        r'(.+)\.adapter_down\.weight$',
        r'(.+)\.adapter_up\.weight$',
    ]

    # NeuroImprint backdoor layer naming patterns
    BACKDOOR_PATTERNS = [
        r'(.*backdoor[^.]*)\.l1\.weight$',
        r'(.*backdoor[^.]*)\.l2\.weight$',
        r'(.*backdoor[^.]*)\.l2\.bias$',
        r'(.*backdoor[^.]*)\.l3\.weight$',
        r'(.*backdoor[^.]*)\.linear1\.weight$',
        r'(.*backdoor[^.]*)\.linear2\.weight$',
        r'(.*backdoor[^.]*)\.linear2\.bias$',
        r'(.*backdoor[^.]*)\.linear3\.weight$',
    ]

    @staticmethod
    def load_from_disk(file_path: str) -> dict[str, torch.Tensor]:
        """Load adapter weights from disk.

        Supports .safetensors, .bin, .pt, .pth files.

        Args:
            file_path: Path to the weight file.

        Returns:
            Dict mapping parameter names to tensors.
        """
        if not file_path.endswith((".safetensors", ".bin", ".pt", ".pth")):
            raise ValueError(
                f"Unsupported file format: {file_path}. "
                "Use .safetensors, .bin, .pt, or .pth"
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Adapter file not found: {file_path}")

        if file_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                return load_file(file_path)
            except ImportError:
                raise ImportError(
                    "safetensors package required for .safetensors files. "
                    "Install with: pip install safetensors"
                )
        elif file_path.endswith((".bin", ".pt", ".pth")):
            data = torch.load(file_path, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                # Handle both raw state_dict and checkpoint dicts
                if "state_dict" in data:
                    return data["state_dict"]
                return data
            raise ValueError(f"Unexpected data format in {file_path}")

        # Should never reach here due to format check above
        return {}

    @staticmethod
    def from_hf_model(model) -> dict[str, torch.Tensor]:
        """Extract state dict from a Hugging Face model.

        Args:
            model: A Hugging Face model (with or without PEFT adapter).

        Returns:
            Dict mapping parameter names to tensors.
        """
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    @staticmethod
    def from_peft_model(peft_model) -> dict[str, torch.Tensor]:
        """Extract only adapter weights from a PEFT model.

        This filters out the base model weights, returning only the
        adapter-specific parameters.

        Args:
            peft_model: A PEFT-wrapped model.

        Returns:
            Dict mapping adapter parameter names to tensors.
        """
        full_sd = {k: v.detach().cpu() for k, v in peft_model.state_dict().items()}
        # Filter to only adapter weights (contain 'lora' or 'adapter' in name)
        adapter_sd = {
            k: v for k, v in full_sd.items()
            if any(marker in k.lower() for marker in ['lora', 'adapter', 'backdoor'])
        }
        return adapter_sd

    def extract_backdoor_candidates(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> list[BackdoorCandidate]:
        """Scan state dict for NeuroImprint backdoor structures.

        Searches for the characteristic 3-layer pattern (L1→L2→L3) with
        the memorization layer L2 being the primary detection target.

        Args:
            state_dict: Dict mapping parameter names to tensors.

        Returns:
            List of BackdoorCandidate objects found.
        """
        candidates: dict[str, BackdoorCandidate] = {}

        for key, tensor in state_dict.items():
            # Try to match backdoor layer patterns
            for pattern in self.BACKDOOR_PATTERNS:
                match = re.match(pattern, key)
                if match:
                    prefix = match.group(1)
                    if prefix not in candidates:
                        candidates[prefix] = BackdoorCandidate(prefix=prefix)

                    candidate = candidates[prefix]

                    # Assign to correct layer
                    if '.l1.weight' in key or '.linear1.weight' in key:
                        candidate.W1 = tensor
                    elif '.l2.weight' in key or '.linear2.weight' in key:
                        candidate.W2 = tensor
                    elif '.l2.bias' in key or '.linear2.bias' in key:
                        candidate.b2 = tensor
                    elif '.l3.weight' in key or '.linear3.weight' in key:
                        candidate.W3 = tensor

                    break

            # Also check for LoRA patterns (backdoor could be disguised as LoRA)
            for pattern in self.LORA_PATTERNS:
                match = re.match(pattern, key)
                if match:
                    prefix = match.group(1)
                    if prefix not in candidates:
                        candidates[prefix] = BackdoorCandidate(prefix=prefix)

                    candidate = candidates[prefix]

                    if 'lora_A' in key:
                        candidate.lora_A = tensor
                    elif 'lora_B' in key:
                        candidate.lora_B = tensor

                    break

        # Filter: only return candidates with at least the memorization layer
        return [
            c for c in candidates.values()
            if c.has_memorization_layer or c.equivalent_W2 is not None
        ]

    def extract_lora_equivalent_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Extract equivalent weight matrices from LoRA adapters.

        For LoRA, the equivalent weight is: W_equivalent = lora_B @ lora_A

        Args:
            state_dict: Dict mapping parameter names to tensors.

        Returns:
            Dict mapping layer names to equivalent weight matrices.
        """
        lora_pairs: dict[str, dict[str, torch.Tensor]] = {}

        for key, tensor in state_dict.items():
            for pattern in self.LORA_PATTERNS:
                match = re.match(pattern, key)
                if match:
                    prefix = match.group(1)
                    if prefix not in lora_pairs:
                        lora_pairs[prefix] = {}

                    if 'lora_A' in key:
                        lora_pairs[prefix]['A'] = tensor
                    elif 'lora_B' in key:
                        lora_pairs[prefix]['B'] = tensor
                    break

        # Compute equivalent weights
        equivalent = {}
        for prefix, pair in lora_pairs.items():
            if 'A' in pair and 'B' in pair:
                # W_equivalent = B @ A (shape: out_features × in_features)
                equivalent[prefix] = pair['B'] @ pair['A']

        return equivalent
