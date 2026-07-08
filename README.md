# lip_reading

CPT (Contrastive Phoneme-Target) Decoder — phoneme-to-text completion with
a decoder-only Llama 3.2 + QLoRA architecture, trained on the LRS2 corpus.

**Run on Google Colab** → [`notebooks/CPT_Decoder_Colab.ipynb`](notebooks/CPT_Decoder_Colab.ipynb)
(single master notebook, Stage-2 full Llama-3.2-3B + 4-bit QLoRA run).

For the uni-PC equivalent (PowerShell, GTX 1080), see [`RUNBOOK_real_run.md`](RUNBOOK_real_run.md).

See [`notebooks/README.md`](notebooks/README.md) for Drive layout, CSV refresh,
and how to reuse the trained adapter on the uni PC.