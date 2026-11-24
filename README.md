# dutch-tutor-dataset

Repository for generating, gating, and validating Dutch tutor synthetic datasets with OpenTaal and Wiktionary lexical grounding.

## Layout

```plaintext
dutch-tutor-dataset/
├── scripts/
│   ├── prepare_repo.py      # Run ONCE locally to build the DB + seed batches
│   ├── validate_batch.py    # Run by Sonnet 5.5 in Arena.ai to gate output
│   ├── preflight_batch.py   # Verbose, read-only diagnostics before/after a gate run
│   └── assemble_batch.py    # Wraps hand-authored rows into the exact training schema
├── assets/
│   └── dutch_lexicon.db     # OpenTaal + Wiktionary SQLite DB (reused in your Android APK)
├── seeds/
│   ├── batch_001.jsonl      # 250 deterministic seeds with pre-anchored [DB_HINTS]
│   └── ...                  # Generated through batch_010.jsonl (2,500 total seeds)
├── data/                    # Sonnet 5.5 writes validated batches here
│   └── .gitkeep
├── .gitignore
└── README.md
```

## Setup & Local Preparation

Run once locally before generating or pushing:

```bash
python scripts/prepare_repo.py --batches 10 --batch-size 250
```

This compiles:
- `assets/dutch_lexicon.db`: Compiled SQLite database (39.1 MB) with OpenTaal official wordlist and Dutch Wiktionary gender/verb data (784,302 words, 100,468 `de`/`het` nouns, and 12,365 verb lemmas).
- `seeds/batch_001.jsonl` through `seeds/batch_010.jsonl`: 2,500 pre-anchored seeds with grammar traps, registers, CEFR levels, and database hints across chat scenarios and OCR explanation scenarios (`ocr_explain`).

## Slice & Repair Batch Validation

Execute deterministic validation inside Arena.ai agent mode for any 250-row slice:

```bash
python scripts/validate_batch.py seeds/batch_001.jsonl data/batch_001.jsonl --start 1 --end 250
```

Validation includes:
- System prompt verification and ID matching
- XML format enforcement (`<reply_nl>`, `<correction>`, `<rule_en>`, `<vocab>`)
- Output length limits (≤ 115 words / 720 characters)
- Mode-specific checks:
  - `valid_dutch`: `<correction>NONE</correction>`
  - `learner_error`: non-NONE correction
  - `ocr_explain`: user prompt must begin with `[OCR_TEXT]` and `<correction>` holds the English translation
- Lexicon compliance (≥ 90% Dutch vocabulary in OpenTaal/Wiktionary) and `de`/`het`/diminutive article agreement
- Compact output of failed IDs for targeted repair prompts

## Authoring Workflow (draft → assemble → gate)

Dutch content is written by hand (never string templates) and only the mechanical
fields are generated, so `preflight_batch.py` can be run before the official gate:

```bash
# 1. Assemble a batch from hand-authored user/reply/correction/rule/vocab fields
python scripts/assemble_batch.py seeds/batch_001.jsonl draft_slice.jsonl data/batch_001.jsonl --append

# 2. Read-only diagnostics: per-row failures plus English-leak and anchor-word hints
python scripts/preflight_batch.py seeds/batch_001.jsonl data/batch_001.jsonl --start 1 --end 250

# 3. Official gate
python scripts/validate_batch.py seeds/batch_001.jsonl data/batch_001.jsonl --start 1 --end 250
```

`preflight_batch.py` adds hints the gate does not print: rows missing every
`anchor_words` entry, possible English tokens leaking into `<reply_nl>`, and
per-row word/character budget. It also reports article conflicts with the full
noun context, which helps separate real errors from false positives such as
`het regent` (verb + pronoun) or `het afval` (the DB stores `afval` as `de`).


## Modal QLoRA Fine-Tuning (A10G)

Train lightweight, on-device Dutch tutor models on Modal using an NVIDIA A10G (24 GB VRAM):

### Anti-Overfitting Safeguards (for 2,000 Samples)
- **Response-Only Loss Masking:** Loss is computed *strictly* on assistant responses (`<reply_nl>...<vocab>`), masking out system prompts and user inputs.
- **Low LoRA Capacity:** $r=16, \alpha=32, \text{dropout}=0.05$ across all projection layers (`q, k, v, o, gate, up, down`).
- **Strict Epoch Cap:** 2 epochs maximum with cosine learning rate decay and 8% warmup.
- **Evaluation Gating:** Automatic 90/10 split (1,800 train / 200 validation) evaluated every 25 steps to save the best checkpoint (`load_best_model_at_end`).

### Commands

```bash
# 1. Prepare combined dataset with 90/10 train/val split
python scripts/prepare_dataset.py

# 2. Launch QLoRA fine-tuning on Modal A10G
# For Qwen (Qwen2.5-0.5B-Instruct / Qwen3-0.6B):
modal run scripts/train_modal.py --model qwen --epochs 2

# For Gemma 3 1B:
modal run scripts/train_modal.py --model gemma --epochs 2

# 3. Download trained weights/adapters from Modal volume to local machine
python scripts/download_modal_weights.py --model qwen --type merged --target-dir models/qwen-dutch-tutor
```
