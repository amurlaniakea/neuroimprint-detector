"""Tests for NeuroImprint Detector."""

import numpy as np
import pytest

from neuroimprint_detector.core.detector import (
    NeuroImprintDetector,
    Verdict,
    AdapterType,
)
from neuroimprint_detector.core.inverter import GradientInverter


class TestNeuroImprintDetector:
    """Test suite for the NeuroImprint detector."""

    def setup_method(self):
        self.detector = NeuroImprintDetector()

    def test_clean_adapter_no_backdoor(self):
        """A random adapter should be classified as clean."""
        weights = {
            'W2': np.random.randn(100, 64),
            'b2': np.random.randn(100),
        }
        result = self.detector.analyze(weights)
        assert result.verdict == Verdict.CLEAN
        assert result.confidence > 0.5

    def test_backdoored_adapter_detected(self):
        """An adapter with NeuroImprint structure should be detected."""
        n_bins = 200
        hidden_dim = 64

        # Create W2 with identical row vectors (backdoor signature)
        r2 = np.random.randn(hidden_dim)
        W2 = np.tile(r2, (n_bins, 1))  # All rows identical

        # Create b2 with sorted interval structure
        b2 = -np.linspace(0.01, 1.0, n_bins)

        weights = {'W2': W2, 'b2': b2}
        result = self.detector.analyze(weights)

        assert result.verdict == Verdict.BACKDOORED
        assert result.confidence >= 0.7
        assert result.has_bin_structure is True
        assert result.has_sorted_biases is True

    def test_suspicious_adapter_partial_signals(self):
        """An adapter with only one signal should be suspicious."""
        n_bins = 100
        hidden_dim = 64

        # Only bin structure, no sorted biases
        r2 = np.random.randn(hidden_dim)
        W2 = np.tile(r2, (n_bins, 1))
        b2 = np.random.randn(n_bins)  # Random, not sorted

        weights = {'W2': W2, 'b2': b2}
        result = self.detector.analyze(weights)

        assert result.verdict in (Verdict.SUSPICIOUS, Verdict.BACKDOORED)

    def test_estimate_samples(self):
        """Should estimate the number of memorized samples from bin count."""
        n_bins = 500
        hidden_dim = 64

        # Create adapter with backdoor structure so detection triggers
        r2 = np.random.randn(hidden_dim)
        W2 = np.tile(r2, (n_bins, 1))
        b2 = -np.linspace(0.01, 1.0, n_bins)

        result = self.detector.analyze({'W2': W2, 'b2': b2})
        assert result.estimated_samples == n_bins

    def test_adapter_type_lora(self):
        """Should identify LoRA adapter type."""
        weights = {'lora_A': np.random.randn(64, 768), 'lora_B': np.random.randn(768, 64)}
        result = self.detector.analyze(weights)
        assert result.adapter_type == AdapterType.LORA

    def test_adapter_type_parallel(self):
        """Should identify parallel adapter type."""
        weights = {'W1': np.random.randn(64, 768), 'W2': np.random.randn(200, 64), 'W3': np.random.randn(768, 200)}
        result = self.detector.analyze(weights)
        assert result.adapter_type == AdapterType.PARALLEL

    def test_bin_structure_score_identical_rows(self):
        """Identical rows should give high bin structure score."""
        r = np.random.randn(64)
        W = np.tile(r, (100, 1))
        score = self.detector._check_bin_structure(W)
        assert score > 0.9

    def test_bin_structure_score_random(self):
        """Random weights should give low bin structure score."""
        np.random.seed(42)  # reproducible
        W = np.random.randn(100, 64)
        score = self.detector._check_bin_structure(W)
        assert score < 0.6  # Random should be around 0.5

    def test_bias_interval_score_sorted(self):
        """Sorted biases should give high interval score."""
        b = np.linspace(-1, 1, 100)
        score = self.detector._check_bias_intervals(b)
        assert score > 0.5

    def test_bias_interval_score_random(self):
        """Random biases should give low interval score."""
        b = np.random.randn(100)
        score = self.detector._check_bias_intervals(b)
        assert score < 0.5


class TestGradientInverter:
    """Test suite for gradient inversion."""

    def setup_method(self):
        self.inverter = GradientInverter()

    def test_inversion_sgd(self):
        """SGD inversion should recover exact gradients."""
        n_bins = 50
        hidden_dim = 64

        W_orig = np.random.randn(n_bins, hidden_dim)
        b_orig = np.random.randn(n_bins)

        # Simulate training: add gradients to weights
        gradients = np.random.randn(n_bins, hidden_dim) * 0.01
        W_trained = W_orig + gradients
        b_trained = b_orig + np.random.randn(n_bins) * 0.01

        result = self.inverter.invert(W_orig, W_trained, b_orig, b_trained, optimizer="sgd")

        assert result.n_samples == n_bins
        assert result.inversion_quality > 0.0
        assert len(result.embeddings) == n_bins

    def test_inversion_adam_approximate(self):
        """Adam inversion should be approximate."""
        n_bins = 50
        hidden_dim = 64

        W_orig = np.random.randn(n_bins, hidden_dim)
        b_orig = np.random.randn(n_bins)

        W_trained = W_orig + np.random.randn(n_bins, hidden_dim) * 0.01
        b_trained = b_orig + np.random.randn(n_bins) * 0.01

        result = self.inverter.invert(W_orig, W_trained, b_orig, b_trained, optimizer="adam")

        assert result.n_samples == n_bins
        # Adam should have lower quality than SGD
        assert any("approximate" in w.lower() for w in result.warnings)

    def test_zero_delta_warning(self):
        """Should warn if no weight difference detected."""
        W = np.random.randn(50, 64)
        b = np.random.randn(50)

        result = self.inverter.invert(W, W.copy(), b, b.copy())

        assert result.n_samples == 0
        assert any("zero" in w.lower() for w in result.warnings)

    def test_shape_mismatch(self):
        """Should handle shape mismatches gracefully."""
        W_orig = np.random.randn(50, 64)
        W_trained = np.random.randn(60, 64)
        b_orig = np.random.randn(50)
        b_trained = np.random.randn(60)

        result = self.inverter.invert(W_orig, W_trained, b_orig, b_trained)

        assert result.n_samples == 0
        assert len(result.warnings) > 0

    def test_embeddings_to_tokens(self):
        """Should map embeddings to token IDs."""
        n_samples = 10
        hidden_dim = 64
        vocab_size = 1000

        embeddings = np.random.randn(n_samples, hidden_dim)
        embedding_matrix = np.random.randn(vocab_size, hidden_dim)

        tokens = self.inverter.embeddings_to_tokens(embeddings, embedding_matrix)

        assert len(tokens) == n_samples
        for token_seq in tokens:
            assert len(token_seq) == 1
            assert 0 <= token_seq[0] < vocab_size
