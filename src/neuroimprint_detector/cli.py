"""CLI for NeuroImprint Detector.

Usage:
    neuroimprint-audit --path /path/to/adapter [--output report.json]
    neuroimprint-audit --model-id meta-llama/Llama-3.2-1B --adapter-path /adapter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from neuroimprint_detector.core.detector import NeuroImprintDetector, Verdict
from neuroimprint_detector.core.inverter import GradientInverter
from neuroimprint_detector.utils.hf_loader import AdapterExtractor


def audit_adapter(
    path: str,
    output: str | None = None,
    verbose: bool = False,
) -> dict:
    """Audit a PEFT adapter for NeuroImprint backdoor.

    Args:
        path: Path to adapter file or directory.
        output: Optional output JSON file path.
        verbose: Print detailed analysis.

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
        # Try to load from directory (HF format)
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
        # Analyze each candidate
        analyses = []
        for candidate in candidates:
            if candidate.has_memorization_layer:
                weights = {
                    'W2': candidate.W2.numpy() if isinstance(candidate.W2, torch.Tensor) else candidate.W2,
                    'b2': candidate.b2.numpy() if isinstance(candidate.b2, torch.Tensor) else candidate.b2,
                }
                det_result = detector.analyze(weights)
                analyses.append({
                    "prefix": candidate.prefix,
                    "verdict": det_result.verdict.value,
                    "confidence": round(det_result.confidence, 4),
                    "has_bin_structure": det_result.has_bin_structure,
                    "has_sorted_biases": det_result.has_sorted_biases,
                    "has_ralu_pattern": det_result.has_ralu_pattern,
                    "estimated_samples": det_result.estimated_samples,
                    "reconstruction_possible": det_result.reconstruction_possible,
                })

        # Overall verdict
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

    # Output
    result_json = json.dumps(result, indent=2)

    if output:
        with open(output, "w") as f:
            f.write(result_json)
        print(f"Report saved to {output}")

    if verbose or not output:
        print(result_json)

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

    args = parser.parse_args()
    audit_adapter(args.path, args.output, args.verbose)


if __name__ == "__main__":
    main()
