"""Tokenizer integration for NeuroImprint text reconstruction.

Maps reconstructed embedding vectors back to human-readable text using
Hugging Face tokenizers.

Supports:
- HF Hub tokenizer IDs (e.g., "meta-llama/Llama-3.2-1B")
- Local tokenizer paths (e.g., "/path/to/tokenizer")
- Graceful fallback when network is unavailable
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


@dataclass
class TokenizerLoadResult:
    """Result of attempting to load a tokenizer."""
    tokenizer: object = None
    embedding_matrix: Optional[np.ndarray] = None
    source: str = ""  # "hf_hub", "local", "failed"
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TextReconstruction:
    """Reconstructed text from embeddings."""
    text: str
    token_ids: list[int]
    confidence: float
    method: str


def _is_local_path(path: str) -> bool:
    """Check if a string is a local filesystem path rather than a HF Hub ID."""
    p = Path(path)
    # If it exists as a directory or file, it's local
    if p.exists():
        return True
    # If it looks like a path (contains / or .) and doesn't look like a HF ID
    # File extensions that indicate local files
    if path.endswith((".json", ".model", ".bin", ".safetensors", ".pt", ".pth")):
        return True
    if "/" in path and not path.startswith("http"):
        if path.startswith("/") or path.startswith(".") or path.endswith((".json", ".model", ".bin")):
            return True
    return False


def load_tokenizer_safe(
    tokenizer_id: str,
    embedding_matrix: Optional[np.ndarray] = None,
) -> TokenizerLoadResult:
    """Load a tokenizer with graceful error handling.

    Tries to load from HF Hub first, then falls back to local path.
    Captures network errors and provides actionable error messages.

    Args:
        tokenizer_id: HF Hub ID or local path to tokenizer directory.
        embedding_matrix: Optional pre-loaded embedding matrix.

    Returns:
        TokenizerLoadResult with tokenizer and status info.
    """
    result = TokenizerLoadResult()

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        result.error = (
            "transformers package required. "
            "Install with: pip install transformers"
        )
        return result

    # Determine if local or HF Hub
    is_local = _is_local_path(tokenizer_id)

    # Try loading tokenizer
    try:
        if is_local:
            if not Path(tokenizer_id).exists():
                result.error = f"Local tokenizer path not found: {tokenizer_id}"
                return result
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)
            result.source = "local"
        else:
            # Try HF Hub (may fail due to network)
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
                result.source = "hf_hub"
            except (OSError, ConnectionError, TimeoutError) as net_err:
                # Network error — suggest local download
                result.warnings.append(
                    f"Could not download tokenizer from HF Hub: {net_err}. "
                    f"Tip: Download the tokenizer locally and use "
                    f"--tokenizer-id /path/to/local/tokenizer"
                )
                result.error = str(net_err)
                return result

        result.tokenizer = tokenizer

    except Exception as e:
        result.error = f"Failed to load tokenizer '{tokenizer_id}': {e}"
        return result

    # Try loading embedding matrix if not provided
    if embedding_matrix is None:
        try:
            if is_local and Path(tokenizer_id).exists():
                model = AutoModel.from_pretrained(tokenizer_id, local_files_only=True)
            else:
                model = AutoModel.from_pretrained(tokenizer_id)

            if hasattr(model, 'get_input_embeddings'):
                emb = model.get_input_embeddings()
                result.embedding_matrix = emb.weight.detach().cpu().numpy()
            else:
                for name, param in model.named_parameters():
                    if 'embed' in name.lower() and 'weight' in name:
                        result.embedding_matrix = param.detach().cpu().numpy()
                        break
                if result.embedding_matrix is None:
                    result.warnings.append("Could not find embedding matrix in model")
        except Exception as e:
            result.warnings.append(f"Could not load embedding matrix: {e}")

    return result


class EmbeddingToText:
    """Converts reconstructed embeddings to text using HF tokenizers."""

    def __init__(
        self,
        tokenizer_id: str = "meta-llama/Llama-3.2-1B",
        embedding_matrix: Optional[np.ndarray] = None,
    ):
        self.tokenizer_id = tokenizer_id
        self._embedding_matrix = embedding_matrix
        self._tokenizer = None
        self._load_warnings: list[str] = []
        self._load_error: Optional[str] = None
        self._load_result: Optional[TokenizerLoadResult] = None

    def _ensure_loaded(self):
        """Lazy-load tokenizer and embedding matrix if not already loaded."""
        if self._load_result is not None:
            return

        self._load_result = load_tokenizer_safe(
            self.tokenizer_id,
            self._embedding_matrix,
        )

        if self._load_result.error:
            self._load_error = self._load_result.error

        if self._load_result.warnings:
            self._load_warnings = self._load_result.warnings

        if self._load_result.tokenizer:
            self._tokenizer = self._load_result.tokenizer

        if self._load_result.embedding_matrix is not None:
            self._embedding_matrix = self._load_result.embedding_matrix

    @property
    def is_available(self) -> bool:
        """Check if the tokenizer is available and ready to use."""
        self._ensure_loaded()
        return self._tokenizer is not None

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def embedding_matrix(self) -> np.ndarray:
        self._ensure_loaded()
        if self._embedding_matrix is None:
            raise ValueError(
                "Embedding matrix not available. "
                "Provide one at initialization or ensure the model can be loaded."
            )
        return self._embedding_matrix

    @property
    def load_warnings(self) -> list[str]:
        self._ensure_loaded()
        return list(self._load_warnings)

    @property
    def load_error(self) -> Optional[str]:
        self._ensure_loaded()
        return self._load_error

    def embeddings_to_text(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        method: str = "nearest",
    ) -> list[TextReconstruction]:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        if not self.is_available:
            return [TextReconstruction(
                text=f"<tokenizer unavailable: {self._load_error}>",
                token_ids=[0],
                confidence=0.0,
                method="failed",
            ) for _ in range(embeddings.shape[0])]

        embeddings = np.asarray(embeddings, dtype=np.float32)
        emb_matrix = self.embedding_matrix

        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norms = np.where(emb_norms == 0, 1, emb_norms)
        embeddings_norm = embeddings / emb_norms

        vocab_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        vocab_norms = np.where(vocab_norms == 0, 1, vocab_norms)
        vocab_norm = emb_matrix / vocab_norms

        similarity = embeddings_norm @ vocab_norm.T

        results = []
        for i in range(embeddings.shape[0]):
            token_ids = [int(np.argmax(similarity[i]))]
            confidence = float(np.max(similarity[i]))

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
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        if not self.is_available:
            return [[TextReconstruction(
                text=f"<tokenizer unavailable: {self._load_error}>",
                token_ids=[0],
                confidence=0.0,
                method="failed",
            )] for _ in range(embeddings.shape[0])]

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
