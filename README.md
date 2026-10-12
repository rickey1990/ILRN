# ILRN-4-96 — Inverse Lag Recurrent Network

**Independent researcher** · rickeyernest1990@gmail.com

ILRN-4-96 is a four-layer, 96-feature network built using four original **ILRM (Inverse Lag Recurrent Memory)** blocks. Each block has separate parameters and recurrent state; the network also includes embeddings, layer normalization, residual connections, and an output head.

## Included code

- `ilrn_reference.py` — four-layer ILRN reference model and ILRMBlock with FFT history.
- `ilrn_fused_cpu.py` — optional CPU fused recurrence and backward pass.
- `verify_ilrn.py` — reference/fused numerical and checkpoint checks.
- `requirements.txt` — Python dependencies.

## Run

```bash
pip install -r requirements.txt
python ilrn_reference.py
python verify_ilrn.py
```

This is a tested reconstruction, **not** the original CNN/DailyMail training program. It does not contain benchmark datasets, and the numerical agreement tests are not independent replications of published benchmark results.

**Licensing:** ILRM documentation and source code are covered by separate noncommercial licences. Refer to `LICENSE_NOTICE.txt` for the exact distinction and links. Incorporating the ILRM mechanism in the ILRN network does not remove those licensing requirements.
