# Changelog

All notable changes to NeuroImprint Detector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-06-22

### Added
- Detector v0.1.0 — Weight analysis for NeuroImprint backdoor signatures (bin structure, bias intervals, RaLU fingerprint)
- Inverter v0.1.0 — Closed-form gradient inversion from adapter weights
- Loader v0.2.0 — HuggingFace adapter loading + backdoor candidate extraction
- Synthetics v0.2.0 — Clean/backdoored adapter generation for testing
- Estimator v0.4.0 — Original weight estimation from trained adapter
- Tokenizer v0.5.0 — Text reconstruction via HF tokenizers (online + offline)
- CLI v0.5.0 — `neuroimprint-audit` command with `--reconstruct`, `--tokenizer-id`, `--output`, `--verbose`
- 43 tests (unit + integration)
- CI/CD via GitHub Actions (Python 3.10, 3.11)
- Makefile with standard targets
- ruff, mypy, black configuration
- Coverage configuration (minimum 80%)
- SECURITY.md and CHANGELOG.md

### Detection Results (from paper)

| Model | Optimizer | Reconstruction Rate | Semantic Similarity |
|-------|-----------|-------------------|-------------------|
| BERT | SGD | 77.4% | 0.994 |
| BERT | AdamW | 74.6% | 0.767 |
| GPT-2 | SGD | 66.5% | 0.990 |
| GPT-2 | AdamW | 74.4% | 0.779 |
| Qwen2-1.5B | SGD | 71.4% | 0.997 |
| Llama3-3B | SGD | 75.0% | 0.997 |
