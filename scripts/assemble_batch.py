#!/usr/bin/env python3
"""Mechanically assemble authored JSONL content into a training batch.

Only structural fields (id / system prompt) are taken from the seed file.
All Dutch and English content is authored by hand in the content JSONL.

Usage:
  python scripts/assemble_batch.py seeds/batch_001.jsonl drafts/slice_001.jsonl data/batch_001.jsonl --append
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("seed_jsonl", type=Path)
    p.add_argument("content_jsonl", type=Path)
    p.add_argument("out_jsonl", type=Path)
    p.add_argument("--append", action="store_true")
    args = p.parse_args()

    seeds = {json.loads(l)["id"]: json.loads(l) for l in args.seed_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()}
    drafts = [json.loads(l) for l in args.content_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = []
    for d in drafts:
        seed = seeds[d["id"]]
        system = (
            f"Je bent een Nederlandse taalcoach. Level: {seed['cefr']}. "
            f"[DB_HINTS: {seed['db_hints']}]"
        )
        assistant = (
            f"<reply_nl>{d['reply_nl']}</reply_nl>\n"
            f"<correction>{d['correction']}</correction>\n"
            f"<rule_en>{d['rule_en']}</rule_en>\n"
            f"<vocab>{d['vocab']}</vocab>"
        )
        rows.append({
            "id": d["id"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": d["user"]},
                {"role": "assistant", "content": assistant},
            ],
        })

    mode = "a" if args.append else "w"
    with args.out_jsonl.open(mode, encoding="utf-8") as fh:
        for row in rows:
            # Compact single-line JSON, no trailing spaces around tags.
            fh.write(json.dumps(row, ensure_ascii=False, separators=(", ", ": ")) + "\n")
    print(f"{'Appended' if args.append else 'Wrote'} {len(rows)} rows to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
