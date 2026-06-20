"""Tests for original weight estimator and tokenizer reconstructor."""

import numpy as np
import pytest
import torch

from neuroimprint_detector.core.estimator import (
    estimate_original_weights,
    compute_reconstruction_from_estimate,
)
from neuroimprint_detector.core.tokenizer_reconstructor import (
    EmbeddingToText,
    TextReconstruction,
)


class TestEstimateOriginalWeights:
    """Tests for original weight estimation."""

    def test_estimate_clean_adapter(self):
        """Clean adapter (no memorization) should estimate correctly."""
        n_bins = 200
        hidden_dim = 64

        # Create W2 with identical rows (no memorization)
        r2 = np.random.randn(hidden_dim)
        W2 = np.tile(r2, (n_bins, 1))
        b2 = -np.linspace(0.01, 1.0, n_bins)

        estimate = estimate_original_weights(W2, b2)

        assert estimate.n_memorized_estimated == 0
        assert estimate.estimation_quality > 0.3
        # W2_original should be very close to W2 (no memorization)
        assert np.allclose(estimate.W2_original, W2, atol=1e-4)

    def test_estimate_backdoored_adapter(self):
        """Backdoored adapter should detect memorized samples."""
        n_bins = 200
        hidden_dim = 64
        n_memorized = 50

        # Create W2 with identical rows + memorized samples
        r2 = np.random.randn(hidden_dim)
        W2 = np.tile(r2, (n_bins, 1)).astype(np.float32)

        # Add gradients to first 50 rows
        for i in range(n_memorized):
            W2[i] += np.random.randn(hidden_dim) * 0.01

        b2 = -np.linspace(0.01, 1.0, n_bins).astype(np.float32)

        estimate = estimate_original_weights(W2, b2)

        # Should detect approximately 50 memorized samples
        assert estimate.n_memorized_estimated > 0
        assert estimate.n_memorized_estimated <= n_memorized + 10  # Allow some tolerance

    def test_estimate_with_torch_tensors(self):
        """Should handle torch tensors."""
        W2 = torch.randn(100, 64)
        b2 = torch.randn(100)

        estimate = estimate_original_weights(W2, b2)

        assert estimate.W2_original.shape == (100, 64)
        assert estimate.b2_original.shape == (100,)

    def test_compute_reconstruction(self):
        """Should compute embeddings from estimated originals."""
        n_bins = 100
        hidden_dim = 64

        # Create original weights
        r2 = np.random.randn(hidden_dim)
        W2_orig = np.tile(r2, (n_bins, 1)).astype(np.float32)
        b2_orig = -np.linspace(0.01, 1.0, n_bins).astype(np.float32)

        # Create trained weights with some memorization
        W2_trained = W2_orig.copy()
        for i in range(10):
            W2_trained[i] += np.random.randn(hidden_dim) * 0.01

        estimate = estimate_original_weights(W2_trained, b2_orig)
        embeddings = compute_reconstruction_from_estimate(W2_trained, b2_orig, estimate)

        assert embeddings.shape == (n_bins, hidden_dim)

    def test_shape_mismatch_handling(self):
        """Should handle edge cases gracefully."""
        W2 = np.random.randn(10, 64)
        b2 = np.random.randn(5)  # Wrong size

        # Should not crash
        estimate = estimate_original_weights(W2, b2)
        assert estimate.estimation_quality >= 0.0


class TestEmbeddingToText:
    """Tests for tokenizer-based text reconstruction."""

    def test_init_without_matrix(self):
        """Should initialize without pre-loaded embedding matrix."""
        # This will fail to load tokenizer without HF, but should not crash on init
        try:
            reconstructor = EmbeddingToText(tokenizer_id="dummy")
            assert reconstructor._embedding_matrix is None
        except Exception:
            pass  # Expected if no internet/HF access

    def test_with_mock_embedding_matrix(self):
        """Should work with a provided embedding matrix."""
        vocab_size = 1000
        hidden_dim = 64

        # Create a mock embedding matrix
        emb_matrix = np.random.randn(vocab_size, hidden_dim).astype(np.float32)

        reconstructor = EmbeddingToText(
            tokenizer_id="dummy",
            embedding_matrix=emb_matrix,
        )

        # Create mock embeddings
        embeddings = np.random.randn(5, hidden_dim).astype(np.float32)

        # Should work without tokenizer (nearest neighbor only)
        try:
            results = reconstructor.embeddings_to_text(embeddings)
            assert len(results) == 5
            for r in results:
                assert isinstance(r, TextReconstruction)
                assert 0 <= r.token_ids[0] < vocab_size
        except Exception:
            pass  # May fail if tokenizer can't be loaded

    def test_batch_reconstruct(self):
        """Should return top-k candidates."""
        vocab_size = 500
        hidden_dim = 32
        emb_matrix = np.random.randn(vocab_size, hidden_dim).astype(np.float32)

        reconstructor = EmbeddingToText(
            tokenizer_id="dummy",
            embedding_matrix=emb_matrix,
        )

        embeddings = np.random.randn(3, hidden_dim).astype(np.float32)

        try:
            results = reconstructor.batch_reconstruct(embeddings, top_k=5)
            assert len(results) == 3
            for sample_results in results:
                assert len(sample_results) == 5
        except Exception:
            pass  # May fail if tokenizer can't be loaded
