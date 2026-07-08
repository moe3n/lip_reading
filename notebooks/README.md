# `notebooks/` — Colab entry point

This directory holds the **single Colab notebook** that runs the full
CPT Decoder training pipeline (Llama-3.2-3B + 4-bit QLoRA on the 48,164-
sentence LRS2 corpus) end-to-end in a fresh Colab runtime.

## Files

- **`CPT_Decoder_Colab.ipynb`** — the master notebook. Open it in Colab
  (`File → Upload notebook` or via `https://colab.research.google.com/github/moe3n/lip_reading/blob/main/notebooks/CPT_Decoder_Colab.ipynb`),
  then run cells top-to-bottom. Total wall time: ~10 min setup + ~3 h
  training on free T4, or ~45 min on Colab Pro A100.

## Google Drive layout (everything Colab writes)

```
/content/drive/MyDrive/cpt_decoder/
├── checkpoints/                    # CPT_CHECKPOINT_DIR (the LoRA adapter lives here)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors   # ~10–25 MB
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── metrics_log.csv             # appended by every run
├── training.log                    # full stdout from the training cell
├── metrics_log.csv                 # mirror of the latest checkpoint's metrics
└── sample_generations.csv          # 10 held-out generations (input → output)
```

The notebook creates `checkpoints/` automatically on first run. Everything
else is created by the corresponding cells.

## Refreshing the LRS2 CSVs

The three CSVs are tracked in this repo at `src/cpt_decoder/data/`:

- `sentphonemepairs_LRS2_original.csv` (8.4 MB, 48,164 rows — headerless LRS2 original)
- `sentences_with_homophones_37374.csv` (1.5 MB)
- `sentences_without_homophones_10790.csv` (307 KB)

To update them:

```bash
cd /content/lip_reading
git pull                          # pulls the latest tracked copies
```

If you maintain a separate upstream of LRS2 phoneme data (e.g. an updated
release), replace the files directly and re-run the notebook from cell 5.
The `_find_data_dir()` walker in `src/cpt_decoder/data/loader.py` resolves
to the first ancestor containing the marker CSV, so placing the files at
`src/cpt_decoder/data/` is sufficient.

## Using the trained adapter back on the uni PC

After the Colab run finishes, copy the adapter folder out of Drive:

1. `MyDrive/cpt_decoder/checkpoints/` → `lip_reading/dryrun_checkpoints/`
2. In Python:
   ```python
   from peft import PeftModel
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from transformers import BitsAndBytesConfig
   bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16)
   base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B",
                  quantization_config=bnb, device_map="auto")
   tok = AutoTokenizer.from_pretrained("dryrun_checkpoints")
   model = PeftModel.from_pretrained(base, "dryrun_checkpoints", is_trainable=False)
   ```
3. Run inference / evaluation per `RUNBOOK_real_run.md`.

## Smoke-testing the notebook (no GPU, no gated token)

Before committing to the 3-hour Stage-2 cell, replace cell 9's
`CPT_MODEL_NAME` with `Qwen/Qwen2.5-0.5B-Instruct` (a non-gated 0.5B
stand-in) and shrink the dataset:

```python
os.environ["CPT_MODEL_NAME"]      = "Qwen/Qwen2.5-0.5B-Instruct"
os.environ["CPT_N_HOMOPHONE"]     = "130"
os.environ["CPT_N_NON_HOMOPHONE"] = "70"
os.environ["CPT_LORA_R"]          = "8"
os.environ["CPT_EPOCHS"]          = "1"
os.environ["CPT_LLM_ERROR_JUDGE"] = "0"
```

This takes ~5 min on a free CPU and proves every cell runs end-to-end
before swapping back to the full Stage-2 settings.