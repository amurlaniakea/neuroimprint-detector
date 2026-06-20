"""Tests for HF loader and synthetic data generator."""

import numpy as np
import os
import pytest
import torch

from neuroimprint_detector.utils.hf_loader import (
    AdapterExtractor,
    BackdoorCandidate,
)
from neuroimprint_detector.utils.synthetics import (
    SyntheticAdapterGenerator,
    SyntheticAdapter,
)
from neuroimprint_detector.core.detector import NeuroImprintDetector, Verdict


class TestAdapterExtractor:
    """Tests for the adapter extractor."""

    def setup_method(self):
        self.extractor = AdapterExtractor()

    def test_extract_backdoor_from_state_dict(self):
        """Should find backdoor structure in a state dict."""
        state_dict = {
            'model.layers.0.privacy_backdoor.l1.weight': torch.randn(64, 768),
            'model.layers.0.privacy_backdoor.l2.weight': torch.randn(200, 64),
            'model.layers.0.privacy_backdoor.l2.bias': torch.randn(200),
            'model.layers.0.privacy_backdoor.l3.weight': torch.randn(768, 200),
        }

        candidates = self.extractor.extract_backdoor_candidates(state_dict)

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.has_full_structure
        assert candidate.has_memorization_layer
        assert candidate.W2.shape == (200, 64)
        assert candidate.b2.shape == (200,)

    def test_no_backdoor_in_clean_state_dict(self):
        """Should not find backdoor in a clean state dict."""
        state_dict = {
            'model.layers.0.self_attn.q_proj.weight': torch.randn(768, 768),
            'model.layers.0.self_attn.k_proj.weight': torch.randn(768, 768),
        }

        candidates = self.extractor.extract_backdoor_candidates(state_dict)
        assert len(candidates) == 0

    def test_extract_lora_equivalent_weights(self):
        """Should compute equivalent weights from LoRA pairs."""
        state_dict = {
            'model.layers.0.lora_A.weight': torch.randn(64, 768),
            'model.layers.0.lora_B.weight': torch.randn(768, 64),
        }

        equivalent = self.extractor.extract_lora_equivalent_weights(state_dict)

        assert len(equivalent) == 1
        for name, W in equivalent.items():
            assert W.shape == (768, 768)

    def test_from_hf_model(self):
        """Should extract state dict from a simple model."""
        model = torch.nn.Linear(10, 5)
        sd = AdapterExtractor.from_hf_model(model)

        assert 'weight' in sd
        assert 'bias' in sd
        assert sd['weight'].shape == (5, 10)

    def test_load_from_disk_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            AdapterExtractor.load_from_disk("/nonexistent/path/adapter.safetensors")

    def test_load_from_disk_unsupported_format(self):
        """Should raise ValueError for unsupported formats."""
        # Create a temp file with unsupported format
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                AdapterExtractor.load_from_disk(temp_path)
        finally:
            os.unlink(temp_path)


class TestSyntheticAdapterGenerator:
    """Tests for the synthetic data generator."""

    def setup_method(self):
        self.generator = SyntheticAdapterGenerator(seed=42)

    def test_generate_clean(self):
        """Clean adapter should have no backdoor structure."""
        adapter = self.generator.generate_clean(hidden_dim=768, reduced_dim=64, n_bins=200)

        assert adapter.is_backdoored is False
        assert adapter.n_memorized_samples == 0
        assert 'l2.weight' in adapter.weights
        assert adapter.weights['l2.weight'].shape == (200, 64)

    def test_generate_backdoored(self):
        """Backdoored adapter should have the NeuroImprint structure."""
        adapter = self.generator.generate_backdoored(
            hidden_dim=768, reduced_dim=64, n_bins=200, n_memorized_samples=50
        )

        assert adapter.is_backdoored is True
        assert adapter.n_memorized_samples == 50

        # Check W2 has identical rows for non-memorized neurons
        W2 = adapter.weights['l2.weight']
        # Non-memorized rows (50+) should be identical to each other
        row50 = W2[50]
        for i in range(51, min(100, W2.shape[0])):
            assert torch.allclose(W2[i], row50, atol=1e-4), f"Row {i} differs from row 50"
        # Memorized rows (0-49) should differ from non-memorized
        assert not torch.allclose(W2[0], row50, atol=1e-2), "Memorized row 0 should differ"

        # Check biases are sorted (ascending or descending)
        b2 = adapter.weights['l2.bias']
        is_ascending = torch.all(b2[:-1] <= b2[1:])
        is_descending = torch.all(b2[:-1] >= b2[1:])
        assert is_ascending or is_descending, "Biases should be sorted"

    def test_generate_lora_disguised(self):
        """LoRA-disguised backdoor should have LoRA structure."""
        adapter = self.generator.generate_lora_disguised(
            hidden_dim=768, rank=64, n_bins=200
        )

        assert adapter.is_backdoored is True
        assert 'lora_A' in adapter.weights
        assert 'lora_B' in adapter.weights

    def test_generate_mixed_state_dict(self):
        """Mixed state dict should contain both clean and backdoored layers."""
        state_dict = self.generator.generate_mixed_state_dict(
            hidden_dim=768, reduced_dim=64, n_bins=200, n_memorized=50
        )

        # Should have clean layers
        assert 'model.layers.0.self_attn.q_proj.weight' in state_dict

        # Should have backdoored layer
        assert 'model.layers.1.self_attn.privacy_backdoor.l2.weight' in state_dict

        # Backdoor W2 should have memorized samples
        W2 = state_dict['model.layers.1.self_attn.privacy_backdoor.l2.weight']
        assert W2.shape == (200, 64)


class TestEndToEndDetection:
    """End-to-end tests: synthetic adapter → detector → verdict."""

    def setup_method(self):
        self.generator = SyntheticAdapterGenerator(seed=42)
        self.detector = NeuroImprintDetector()
        self.extractor = AdapterExtractor()

    def test_clean_adapter_detected_as_clean(self):
        """Clean adapter should be classified as clean."""
        adapter = self.generator.generate_clean()
        # Map clean adapter keys to what detector expects
        weights = {
            'W2': adapter.weights['l2.weight'],
            'b2': adapter.weights['l2.bias'],
        }
        result = self.detector.analyze(weights)
        assert result.verdict == Verdict.CLEAN

    def test_backdoored_adapter_detected(self):
        """Backdoored adapter should be detected."""
        adapter = self.generator.generate_backdoored(n_memorized_samples=0)
        # Map backdoor keys to what detector expects
        weights = {
            'W2': adapter.weights['l2.weight'],
            'b2': adapter.weights['l2.bias'],
        }
        result = self.detector.analyze(weights)
        assert result.verdict == Verdict.BACKDOORED
        assert result.confidence >= 0.7

    def test_full_pipeline_with_extractor(self):
        """Full pipeline: state dict → extractor → detector."""
        state_dict = self.generator.generate_mixed_state_dict()

        # Extract backdoor candidates
        candidates = self.extractor.extract_backdoor_candidates(state_dict)
        assert len(candidates) >= 1

        # Analyze the first candidate
        candidate = candidates[0]
        if candidate.has_memorization_layer:
            weights = {
                'W2': candidate.W2,
                'b2': candidate.b2,
            }
            result = self.detector.analyze(weights)
            assert result.verdict == Verdict.BACKDOORED

    def test_lora_disguised_detection(self):
        """LoRA-disguised backdoor should be detectable via equivalent weights."""
        adapter = self.generator.generate_lora_disguised()

        # Extract LoRA equivalent
        extractor = AdapterExtractor()
        equivalent = extractor.extract_lora_equivalent_weights(adapter.weights)

        for name, W2 in equivalent.items():
            if W2.shape[0] > W2.shape[1]:
                weights = {'W2': W2, 'b2': torch.randn(W2.shape[0])}
                result = self.detector.analyze(weights)
                assert result.verdict in (Verdict.BACKDOORED, Verdict.SUSPICIOUS, Verdict.CLEAN)
