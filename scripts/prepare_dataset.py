#!/usr/bin/env python3
"""
Prepares the full 2,000-row dataset for fine-tuning by merging validated batches
and creating a reproducible 90/10 train/val split.

Usage:
  python scripts/prepare_dataset.py
"""

import glob
import json
import random
from pathlib import Path

DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
COMBINED_FILE = DATA_DIR / "dataset_2000.jsonl"


def main():
    batch_files = sorted(DATA_DIR.glob("batch_*.jsonl"))
    if not batch_files:
        print("FAIL: No batch files found in data/")
        return 1

    all_rows = []
    seen_ids = set()
    for bf in batch_files:
        with bf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = row.get("id")
                if sid in seen_ids:
                    print(f"Warning: duplicate id {sid} in {bf.name}")
                seen_ids.add(sid)
                all_rows.append(row)

    print(f"Loaded {len(all_rows)} total rows from {len(batch_files)} batches.")

    # Write combined file
    with COMBINED_FILE.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote combined dataset ({len(all_rows)} rows) -> {COMBINED_FILE}")

    # Reproducible split (90% train / 10% val)
    rng = random.Random(42)
    shuffled = list(all_rows)
    rng.shuffle(shuffled)

    val_count = int(len(shuffled) * 0.10)  # 200 rows
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]     # 1,800 rows

    with TRAIN_FILE.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with VAL_FILE.open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Split complete: {len(train_rows)} train rows -> {TRAIN_FILE}, {len(val_rows)} val rows -> {VAL_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
