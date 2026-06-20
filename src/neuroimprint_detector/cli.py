"""CLI for NeuroImprint Detector.

Usage:
    neuroimprint-audit --path /path/to/adapter [--output report.json] [--tokenizer-id meta-llama/Llama-3.2-1B]
    neuroimprint-audit --path /adapter --reconstruct --tokenizer-id Qwen/Qwen2-0.5B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from neuroimprint_detector.core.detector import NeuroImprintDetector, Verdict
from neuroimprint_detector.core.estimator import (
    estimate_original_weights,
    compute_reconstruction_from_estimate,
)
from neuroimprint_detector.utils.hf_loader import AdapterExtractor


def audit_adapter(
    path: str,
    output: str | None = None,
    verbose: bool = False,
    reconstruct: bool = False,
    tokenizer_id: str | None = None,
) -> dict:
    """Audit a PEFT adapter for NeuroImprint backdoor.

    Args:
        path: Path to adapter file or directory.
        output: Optional output JSON file path.
        verbose: Print detailed analysis.
        reconstruct: Attempt to reconstruct memorized samples.
        tokenizer_id: HF tokenizer ID for text reconstruction.

    Returns:
        Dict with audit results.
    """
    path = Path(path)
    extractor = AdapterExtractor()
    detector = NeuroImprintDetector()

    # Load adapter
    if path.is_file():
        state_dict = AdapterExtractor.load_from_disk(str(path))
    elif path.is_dir():
        safetensors_path = path / "adapter_model.safetensors"
        bin_path = path / "adapter_model.bin"
        if safetensors_path.exists():
            state_dict = AdapterExtractor.load_from_disk(str(safetensors_path))
        elif bin_path.exists():
            state_dict = AdapterExtractor.load_from_disk(str(bin_path))
        else:
            print(f"Error: No adapter file found in {path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Extract backdoor candidates
    candidates = extractor.extract_backdoor_candidates(state_dict)

    if not candidates:
        result = {
            "verdict": "clean",
            "confidence": 0.9,
            "n_candidates": 0,
            "message": "No NeuroImprint backdoor structure detected",
        }
    else:
        analyses = []
        for candidate in candidates:
            if candidate.has_memorization_layer:
                W2 = candidate.W2.numpy() if isinstance(candidate.W2, torch.Tensor) else candidate.W2
                b2 = candidate.b2.numpy() if isinstance(candidate.b2, torch.Tensor) else candidate.b2

                det_result = detector.analyze({'W2': W2, 'b2': b2})

                analysis = {
                    "prefix": candidate.prefix,
                    "verdict": det_result.verdict.value,
                    "confidence": round(det_result.confidence, 4),
                    "has_bin_structure": det_result.has_bin_structure,
                    "has_sorted_biases": det_result.has_sorted_biases,
                    "has_ralu_pattern": det_result.has_ralu_pattern,
                    "estimated_samples": det_result.estimated_samples,
                    "reconstruction_possible": det_result.reconstruction_possible,
                }

                # Reconstruct if requested and possible
                if reconstruct and det_result.verdict in (Verdict.BACKDOORED, Verdict.SUSPICIOUS):
                    reconstruction = _reconstruct_samples(
                        W2, b2, tokenizer_id, verbose
                    )
                    analysis["reconstruction"] = reconstruction

                analyses.append(analysis)

        backdoored = [a for a in analyses if a["verdict"] == "backdoored"]
        suspicious = [a for a in analyses if a["verdict"] == "suspicious"]

        if backdoored:
            verdict = "backdoored"
            confidence = max(a["confidence"] for a in backdoored)
        elif suspicious:
            verdict = "suspicious"
            confidence = max(a["confidence"] for a in suspicious)
        else:
            verdict = "clean"
            confidence = 0.9

        result = {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "n_candidates": len(candidates),
            "analyses": analyses,
        }

    result_json = json.dumps(result, indent=2, default=str)

    if output:
        with open(output, "w") as f:
            f.write(result_json)
        print(f"Report saved to {output}")

    if verbose or not output:
        print(result_json)

    return result


def _reconstruct_samples(
    W2: "np.ndarray | torch.Tensor",
    b2: "np.ndarray | torch.Tensor",
    tokenizer_id: str | None,
    verbose: bool,
) -> dict:
    """Reconstruct memorized samples from adapter weights.

    Uses original weight estimation + gradient inversion + optional
    tokenizer-based text reconstruction.
    """
    import numpy as np

    if isinstance(W2, torch.Tensor):
        W2 = W2.detach().cpu().numpy()
    if isinstance(b2, torch.Tensor):
        b2 = b2.detach().cpu().numpy()

    # Estimate original weights
    estimate = estimate_original_weights(W2, b2)

    # Compute reconstructed embeddings
    embeddings = compute_reconstruction_from_estimate(W2, b2, estimate)

    result = {
        "n_memorized_estimate": estimate.n_memorized_estimated,
        "estimation_quality": round(estimate.estimation_quality, 4),
        "n_embeddings": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "warnings": estimate.warnings,
    }

    # Text reconstruction with tokenizer
    if tokenizer_id and embeddings.shape[0] > 0:
        try:
            from neuroimprint_detector.core.tokenizer_reconstructor import load_tokenizer_safe
            tok_result = load_tokenizer_safe(tokenizer_id)
            if tok_result.tokenizer is not None:
                from neuroimprint_detector.core.tokenizer_reconstructor import EmbeddingToText
                reconstructor = EmbeddingToText(
                    tokenizer_id=tokenizer_id,
                    embedding_matrix=tok_result.embedding_matrix,
                )
                if reconstructor.is_available:
                    texts = reconstructor.embeddings_to_text(embeddings[:10])
                    result["reconstructed_texts"] = [
                        {"text": t.text, "confidence": round(t.confidence, 4)}
                        for t in texts
                    ]
                else:
                    result["tokenizer_warning"] = reconstructor.load_error
            else:
                result["tokenizer_warning"] = tok_result.error
        except Exception as e:
            result["tokenizer_error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="NeuroImprint Detector — Audit PEFT adapters for privacy backdoors"
    )
    parser.add_argument(
        "--path", required=True,
        help="Path to adapter file (.safetensors, .bin) or directory"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON report file path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed analysis"
    )
    parser.add_argument(
        "--reconstruct", action="store_true",
        help="Attempt to reconstruct memorized samples"
    )
    parser.add_argument(
        "--tokenizer-id", default=None,
        help="Hugging Face tokenizer ID for text reconstruction (e.g., meta-llama/Llama-3.2-1B)"
    )

    args = parser.parse_args()
    audit_adapter(
        args.path, args.output, args.verbose,
        args.reconstruct, args.tokenizer_id,
    )


if __name__ == "__main__":
    main()
