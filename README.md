# ILRN-4-96 — Inverse Lag Recurrent Network

**Attention-free AI · Recurrent neural network (RNN) · Inverse-lag memory · FFT-accelerated sequence modeling · PyTorch**

Independent researcher · **rickeyernest1990@gmail.com**

**ILRN** is an experimental attention-free recurrent neural-network architecture built from **ILRM (Inverse Lag Recurrent Memory)** blocks. **ILRN-4-96** uses four independently parameterized ILRM blocks with a 96-dimensional feature width, layer normalization, residual connections, token embeddings, and an untied prediction head.

Instead of Transformer-style self-attention, each ILRM block combines a **normalized harmonic inverse-lag (1/k) history** with an **access gate**, **write gate**, and recurrent hidden state. The history can be calculated by FFT-based causal convolution. An optional **fused CPU kernel** reduces recurrent execution overhead without changing the intended ILRM equations.

> **Naming:** ILRM is the original memory mechanism; ILRN is the network composed of ILRM blocks. “Attentionless AI” is an informal search term; **attention-free** or **without self-attention** is the preferred technical description.

## Architecture

- **Network:** ILRN-4-96 (four ILRM blocks, hidden width 96)
- **Memory mechanism:** ILRM with causal harmonic (inverse-lag) history and two recurrent gates
- **History calculation:** zero-lag-excluding FFT causal convolution; no current-token leakage
- **Output:** vocabulary logits for next-token prediction
- **Execution:** PyTorch reference; optional Numba-compiled fused **CPU float32** recurrence
- **Evaluation:** numerical parity checks for reference and fused implementations

The architecture is **attention-free** but not “without memory”: it keeps both the harmonic input history and the learned recurrent state.

## Code in this repository

| File | Purpose |
| --- | --- |
| `ilrn_reference.py` | ILRMBlock recurrence, FFT inverse-lag history, and the four-layer `StackedILRN` network |
| `ilrn_fused_cpu.py` | Optional fused CPU recurrent forward/backward execution |
| `verify_ilrn.py` | Reference-vs-fused forward, gradient, and one-step update checks |
| `requirements.txt` | Required Python packages |
| `VALIDATION.txt` | Previously recorded numerical check results |
| `LICENSE_NOTICE.txt` | Original ILRM documentation/code licensing distinction |
| `README_ILRN_Code.txt` | Additional code usage notes |

## Try it

```bash
pip install -r requirements.txt
python ilrn_reference.py
python verify_ilrn.py
```

The example code is a tested **reconstruction**, not the original CNN/DailyMail experiment runner. It does not include the benchmark training datasets or trained checkpoints. Numerical parity on test inputs does **not** independently reproduce published benchmark results, establish general superiority, or guarantee a universal speed-up.

## Research terminology / searchable keywords

`attention-free-ai` · `attentionless-ai` · `recurrent-neural-network` · `rnn` · `inverse-lag-memory` · `harmonic-memory` · `long-context` · `fft` · `pytorch` · `deep-learning` · `sequence-modeling` · `language-model` · `numba` · `cpu-optimization`

## ILRM licensing and ILRN attribution

ILRM is the original licensed memory component used in the ILRN layers:

- **ILRM documentation, explanatory text, and figures:** Creative Commons Attribution–NonCommercial 4.0 (CC BY-NC 4.0).
- **ILRM implementation code and source listings:** PolyForm Noncommercial License 1.0.0.

See [LICENSE_NOTICE.txt](LICENSE_NOTICE.txt). These licences apply to their relevant ILRM materials; this README does **not** declare a new blanket licence over all separately created ILRN contributions. Including ILRM blocks within ILRN does not waive relevant ILRM licensing conditions.
