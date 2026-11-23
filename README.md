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
│   ├── batch_001.jsonl      # 100 deterministic seeds with pre-anchored [DB_HINTS]
│   └── ...
└── data/                    # Sonnet 5.5 writes validated batches here
    └── .gitkeep
```

## Setup & Local Preparation

Run once locally before generating or pushing:

```bash
python scripts/prepare_repo.py --batches 25 --batch-size 100
```

This compiles:
- `assets/dutch_lexicon.db`: Compiled SQLite database with OpenTaal official wordlist and Dutch Wiktionary gender/verb data.
- `seeds/batch_001.jsonl` through `seeds/batch_025.jsonl`: 2,500 pre-anchored seeds with grammar traps, registers, CEFR levels, and database hints.

## Batch Validation

Execute deterministic validation inside Arena.ai agent mode:

```bash
python scripts/validate_batch.py seeds/batch_001.jsonl data/batch_001.jsonl
```
