"""Tokenizer integration for NeuroImprint text reconstruction.

Maps reconstructed embedding vectors back to human-readable text using
Hugging Face tokenizers.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union

import torch


@dataclass
class TextReconstruction:
    """Reconstructed text from embeddings."""
    text: str
    token_ids: list[int]
    confidence: float
    method: str  # "nearest", "beam", "greedy"


class EmbeddingToText:
    """Converts reconstructed embeddings to text using HF tokenizers.

    Supports:
    - Nearest-neighbor token lookup (fast, approximate)
    - Batch processing for multiple samples
    - Confidence scoring per token
    """

    def __init__(
        self,
        tokenizer_id: str = "meta-llama/Llama-3.2-1B",
        embedding_matrix: Optional[np.ndarray] = None,
    ):
        """Initialize with a tokenizer and optional embedding matrix.

        Args:
            tokenizer_id: Hugging Face tokenizer identifier.
            embedding_matrix: Optional pre-loaded embedding matrix.
                If None, will be loaded from the tokenizer's model.
        """
        self.tokenizer_id = tokenizer_id
        self._tokenizer = None
        self._embedding_matrix = embedding_matrix

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
            except ImportError:
                raise ImportError(
                    "transformers package required. "
                    "Install with: pip install transformers"
                )
        return self._tokenizer

    @property
    def embedding_matrix(self) -> np.ndarray:
        """Get the embedding matrix (lazy-loaded)."""
        if self._embedding_matrix is None:
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(self.tokenizer_id)
                # Get the input embedding matrix
                if hasattr(model, 'get_input_embeddings'):
                    emb = model.get_input_embeddings()
                    self._embedding_matrix = emb.weight.detach().cpu().numpy()
                else:
                    # Fallback: try to find embedding layer
                    for name, param in model.named_parameters():
                        if 'embed' in name.lower() and 'weight' in name:
                            self._embedding_matrix = param.detach().cpu().numpy()
                            break
                    if self._embedding_matrix is None:
                        raise ValueError("Could not find embedding matrix in model")
            except ImportError:
                raise ImportError(
                    "transformers package required. "
                    "Install with: pip install transformers"
                )
        return self._embedding_matrix

    def embeddings_to_text(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        method: str = "nearest",
    ) -> list[TextReconstruction]:
        """Convert embedding vectors to text.

        Args:
            embeddings: Embedding vectors (n_samples, hidden_dim)
            method: "nearest" for nearest-neighbor lookup

        Returns:
            List of TextReconstruction objects.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        embeddings = np.asarray(embeddings, dtype=np.float32)
        emb_matrix = self.embedding_matrix

        # Normalize for cosine similarity
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norms = np.where(emb_norms == 0, 1, emb_norms)
        embeddings_norm = embeddings / emb_norms

        vocab_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        vocab_norms = np.where(vocab_norms == 0, 1, vocab_norms)
        vocab_norm = emb_matrix / vocab_norms

        # Cosine similarity: (n_samples, vocab_size)
        similarity = embeddings_norm @ vocab_norm.T

        results = []
        for i in range(embeddings.shape[0]):
            if method == "nearest":
                token_ids = [int(np.argmax(similarity[i]))]
                confidence = float(np.max(similarity[i]))
            else:
                token_ids = [int(np.argmax(similarity[i]))]
                confidence = float(np.max(similarity[i]))

            # Decode to text
            try:
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception:
                text = f"<token_{token_ids[0]}>"

            results.append(TextReconstruction(
                text=text,
                token_ids=token_ids,
                confidence=confidence,
                method=method,
            ))

        return results

    def batch_reconstruct(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> list[list[TextReconstruction]]:
        """Reconstruct text with top-k candidates per sample.

        Args:
            embeddings: Embedding vectors (n_samples, hidden_dim)
            top_k: Number of top candidates to return per sample.

        Returns:
            List of lists of TextReconstruction (one list per sample).
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        embeddings = np.asarray(embeddings, dtype=np.float32)
        emb_matrix = self.embedding_matrix

        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norms = np.where(emb_norms == 0, 1, emb_norms)
        embeddings_norm = embeddings / emb_norms

        vocab_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        vocab_norms = np.where(vocab_norms == 0, 1, vocab_norms)
        vocab_norm = emb_matrix / vocab_norms

        similarity = embeddings_norm @ vocab_norm.T

        all_results = []
        for i in range(embeddings.shape[0]):
            top_indices = np.argsort(similarity[i])[-top_k:][::-1]
            sample_results = []
            for idx in top_indices:
                token_ids = [int(idx)]
                try:
                    text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                except Exception:
                    text = f"<token_{token_ids[0]}>"
                sample_results.append(TextReconstruction(
                    text=text,
                    token_ids=token_ids,
                    confidence=float(similarity[i][idx]),
                    method="top_k",
                ))
            all_results.append(sample_results)

        return all_results
