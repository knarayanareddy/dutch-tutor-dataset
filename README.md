# dutch-tutor-dataset

Repository for generating, gating, and validating Dutch tutor synthetic datasets with OpenTaal and Wiktionary lexical grounding.

## Layout

```plaintext
dutch-tutor-dataset/
├── scripts/
│   ├── prepare_repo.py      # Run ONCE locally to build the DB + seed batches
│   └── validate_batch.py    # Run by Sonnet 5.5 in Arena.ai to gate output
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
