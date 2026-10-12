ILRN-4-96 NETWORK / ILRM MEMORY BLOCK REPRODUCTION CODE
12 October 2026 | Independent researcher | rickeyernest1990@gmail.com

ILRN-4-96 is a four-layer 96-feature network made using four ILRM (Inverse
Lag Recurrent Memory) blocks. Each ILRM block has its own parameters and
separate recurrent state; the network supplies embeddings and an output head.
The ILRM documentation and software are subject to the split noncommercial
licences described in LICENSE_NOTICE.txt.

Source files:
  ilrn_reference.py    four-layer ILRN model; ILRMBlock cell and FFT history
  ilrn_fused_cpu.py    optional fused CPU recurrence / backward implementation
  verify_ilrn.py       numerical and checkpoint checks
  requirements.txt     Python dependencies

After installing dependencies, run:
  python ilrn_reference.py
  python verify_ilrn.py

The source code is a tested reconstruction, not the original data-training
script, and does not include the external benchmark datasets. Numeric
checks on sampled CPU inputs are not independent reproductions of published
news-model or fused-kernel benchmark scores.

See LICENSE_NOTICE.txt before redistributing or adapting the source code.
