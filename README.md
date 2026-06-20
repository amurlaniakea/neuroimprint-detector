# NeuroImprint Detector

Audit PEFT adapters for privacy backdoors in federated learning.

**Author:** Pedro Sordo Martínez (amurlaniakea@gmail.com)
**License:** AGPL-3.0-or-later
**Status:** v0.2.0 — Adapter loader + synthetic data, 29 tests passing

## Overview

NeuroImprint (Shi et al., 2026) is a privacy attack against federated learning that corrupts PEFT adapters to memorize client training samples. A malicious parameter server can reconstruct 59-79% of training data from the adapter weights alone.

This project implements the **detector**: given a PEFT adapter, determine if it contains a NeuroImprint backdoor and reconstruct memorized samples.

## The Attack

1. Attacker inserts a backdoor in a parallel PEFT adapter with 3 layers:
   - **Projection (L1):** Reduces embedding dimension (h → ĥ)
   - **Memorization (L2):** Neurons organized as "reconstruction bins" — one neuron per sample
   - **Output (L3):** Maps back, canceled by LayerNorm (invisible)

2. Each training sample activates exactly one neuron (linear activation via RaLU)

3. After training, the attacker inverts the weights:
   ```
   x̃ = (W̃₂ - W₂) / (b̃₂ - b₂)
   ```

4. Reconstructed embeddings are mapped back to token sequences

## Detection Strategy

The detector analyzes adapter weights for the NeuroImprint signature:

1. **Bin structure:** W₂ has identical row vectors (r₂ repeated)
2. **Bias intervals:** b₂ is sorted in ascending intervals
3. **RaLU fingerprint:** Rank-1 weight matrix from reduced activation range

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Usage

```python
from neuroimprint_detector import NeuroImprintDetector, GradientInverter
import numpy as np

# Analyze an adapter
detector = NeuroImprintDetector()
result = detector.analyze({
    'W2': adapter_W2,  # Weight matrix
    'b2': adapter_b2,  # Bias vector
})

print(f"Verdict: {result.verdict.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Estimated samples: {result.estimated_samples}")

# If backdoored, reconstruct samples
if result.reconstruction_possible:
    inverter = GradientInverter()
    inversion = inverter.invert(
        W_original, W_trained,
        b_original, b_trained,
        optimizer="sgd"
    )
    print(f"Reconstructed {inversion.n_samples} samples")
    print(f"Quality: {inversion.inversion_quality:.2f}")
```

## Results (from paper)

| Model | Optimizer | Reconstruction Rate | Semantic Similarity |
|-------|-----------|-------------------|-------------------|
| BERT | SGD | 77.4% | 0.994 |
| BERT | AdamW | 74.6% | 0.767 |
| GPT-2 | SGD | 66.5% | 0.990 |
| GPT-2 | AdamW | 74.4% | 0.779 |
| Qwen2-1.5B | SGD | 71.4% | 0.997 |
| Llama3-3B | SGD | 75.0% | 0.997 |

## References

- **Shi et al. (2026)** — [From Efficiency to Leakage: Privacy Backdoor in Federated Language Model Fine-Tuning](https://arxiv.org/abs/2606.20553)
- **Hu et al. (2022)** — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Zhu et al. (2019)** — [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935)
