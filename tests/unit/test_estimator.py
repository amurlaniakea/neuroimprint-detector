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
    load_tokenizer_safe,
    _is_local_path,
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
        reconstructor = EmbeddingToText(tokenizer_id="dummy")
        assert reconstructor._embedding_matrix is None

    def test_is_available_with_mock_matrix(self):
        """Should report available when embedding matrix is provided."""
        emb_matrix = np.random.randn(100, 64).astype(np.float32)
        reconstructor = EmbeddingToText(
            tokenizer_id="dummy",
            embedding_matrix=emb_matrix,
        )
        # Without a real tokenizer, is_available depends on load
        # but should not crash
        _ = reconstructor.is_available

    def test_with_mock_embedding_matrix(self):
        """Should work with a provided embedding matrix."""
        vocab_size = 1000
        hidden_dim = 64
        emb_matrix = np.random.randn(vocab_size, hidden_dim).astype(np.float32)

        reconstructor = EmbeddingToText(
            tokenizer_id="dummy",
            embedding_matrix=emb_matrix,
        )

        embeddings = np.random.randn(5, hidden_dim).astype(np.float32)

        results = reconstructor.embeddings_to_text(embeddings)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, TextReconstruction)
            assert 0 <= r.token_ids[0] < vocab_size

    def test_batch_reconstruct(self):
        """Should return top-k candidates (or graceful failure without tokenizer)."""
        vocab_size = 500
        hidden_dim = 32
        emb_matrix = np.random.randn(vocab_size, hidden_dim).astype(np.float32)

        reconstructor = EmbeddingToText(
            tokenizer_id="dummy",
            embedding_matrix=emb_matrix,
        )

        embeddings = np.random.randn(3, hidden_dim).astype(np.float32)

        results = reconstructor.batch_reconstruct(embeddings, top_k=5)
        assert len(results) == 3
        for sample_results in results:
            # Each sample should have at least 1 result (may be <5 if tokenizer unavailable)
            assert len(sample_results) >= 1


class TestLoadTokenizerSafe:
    """Tests for safe tokenizer loading."""

    def test_is_local_path_absolute(self):
        assert _is_local_path("/path/to/tokenizer") is True

    def test_is_local_path_relative(self):
        assert _is_local_path("./tokenizer") is True

    def test_is_local_path_with_extension(self):
        assert _is_local_path("tokenizer.json") is True

    def test_is_local_path_hf_id(self):
        assert _is_local_path("meta-llama/Llama-3.2-1B") is False

    def test_load_nonexistent_local_path(self):
        result = load_tokenizer_safe("/nonexistent/path/tokenizer")
        assert result.error is not None
        assert result.tokenizer is None

