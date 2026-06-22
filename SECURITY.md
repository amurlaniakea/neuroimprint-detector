# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.5.0   | Yes       |

## Reporting a Security Vulnerability

If you discover a security vulnerability in NeuroImprint Detector, please report it responsibly.

**Do NOT open a public GitHub Issue for security vulnerabilities.**

Instead, report via email:
- **Email:** amurlaniakea@gmail.com
- **Subject:** `[SECURITY] NeuroImprint Detector vulnerability`

You will receive a response within 48 hours.

## Security Considerations

NeuroImprint Detector is a forensic tool for auditing PEFT adapters:

- **Detection accuracy** depends on the quality of the weight analysis. Sophisticated attackers may design backdoors that evade the current detection heuristics.
- **Reconstruction** is only possible when the NeuroImprint signature is present. Clean adapters will not produce meaningful reconstructions.
- **HF Hub integration** requires network access. Use `--tokenizer-id /path/to/local/tokenizer` for offline operation.
- **torch dependency** is heavy. Consider CPU-only installs for CI environments: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Dependencies

Runtime: `numpy`, `torch`, `transformers`, `peft`
Dev: `pytest`, `pytest-cov`
