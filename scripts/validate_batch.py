#!/usr/bin/env python3
"""
Deterministic validation gate for Dutch tutor JSONL batches (including ocr_explain mode).
Usage:
  python scripts/validate_batch.py seeds/batch_001.jsonl data/batch_001.jsonl --start 1 --end 250
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("assets/dutch_lexicon.db")

TAG_PATTERN = re.compile(
    r"^<reply_nl>(?P<reply_nl>.*?)</reply_nl>\s*"
    r"<correction>(?P<correction>.*?)</correction>\s*"
    r"<rule_en>(?P<rule_en>.*?)</rule_en>\s*"
    r"<vocab>(?P<vocab>.*?)</vocab>$",
    re.DOTALL,
)
ARTICLE_NOUN_RE = re.compile(r"\b(de|het)\s+([a-zàáâäèéêëìíîïòóôöùúûüĳ'-]{3,})\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zàáâäèéêëìíîïòóôöùúûüĳ'-]+", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--start", type=int, default=1, help="1-based start row in seed file")
    parser.add_argument("--end", type=int, default=None, help="1-based inclusive end row in seed file")
    args = parser.parse_args()

    all_seeds = [
        json.loads(line)
        for line in args.seed_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    end_idx = args.end if args.end is not None else len(all_seeds)
    seeds = all_seeds[args.start - 1 : end_idx]

    if not args.output_jsonl.exists():
        print(f"FAIL: {args.output_jsonl} does not exist.")
        return 1

    out_lines = [
        line for line in args.output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(seeds) != len(out_lines):
        print(f"FAIL: Expected {len(seeds)} rows (seeds {args.start}..{end_idx}), got {len(out_lines)}.")
        return 1

    conn = sqlite3.connect(DB_PATH)
    valid_words = {r[0] for r in conn.execute("SELECT word FROM words")}
    noun_articles = {r[0]: r[1] for r in conn.execute("SELECT lemma, article FROM nouns")}
    conn.close()

    errors = []
    failed_ids = []
    for idx, (seed, raw_line) in enumerate(zip(seeds, out_lines), start=args.start):
        sid = seed["id"]
        try:
            row = json.loads(raw_line)
            msgs = row["messages"]
            assert row.get("id") == sid
            assert len(msgs) == 3
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user" and msgs[1]["content"].strip()
            assert msgs[2]["role"] == "assistant"
        except Exception as e:
            errors.append(f"Row {idx} ({sid}): invalid JSON/id/messages ({e})")
            failed_ids.append(sid)
            continue

        expected_sys = f"Je bent een Nederlandse taalcoach. Level: {seed['cefr']}. [DB_HINTS: {seed['db_hints']}]"
        if msgs[0]["content"].strip() != expected_sys:
            errors.append(f"Row {idx} ({sid}): system prompt mismatch")
            failed_ids.append(sid)
            continue

        user_msg = msgs[1]["content"].strip()
        asst = msgs[2]["content"].strip()
        m = TAG_PATTERN.match(asst)
        if not m:
            errors.append(f"Row {idx} ({sid}): missing or malformed XML tags")
            failed_ids.append(sid)
            continue

        if len(asst.split()) > 115 or len(asst) > 720:
            errors.append(f"Row {idx} ({sid}): too long ({len(asst.split())} words)")
            failed_ids.append(sid)
            continue

        reply_nl = m.group("reply_nl").strip()
        correction = m.group("correction").strip()
        rule_en = m.group("rule_en").strip()
        vocab = m.group("vocab").strip()

        if not reply_nl or not rule_en or not vocab:
            errors.append(f"Row {idx} ({sid}): empty tag")
            failed_ids.append(sid)
            continue

        mode = seed["mode"]
        if mode == "valid_dutch" and correction.upper() != "NONE":
            errors.append(f"Row {idx} ({sid}): valid_dutch must have NONE correction")
            failed_ids.append(sid)
            continue
        if mode == "learner_error" and correction.upper() == "NONE":
            errors.append(f"Row {idx} ({sid}): learner_error cannot have NONE correction")
            failed_ids.append(sid)
            continue
        if mode == "ocr_explain":
            if not user_msg.startswith("[OCR_TEXT]"):
                errors.append(f"Row {idx} ({sid}): ocr_explain user message must start with '[OCR_TEXT]'")
                failed_ids.append(sid)
                continue
            if correction.upper() == "NONE" or len(correction.split()) < 2:
                errors.append(f"Row {idx} ({sid}): ocr_explain requires English translation in <correction>")
                failed_ids.append(sid)
                continue

        # In ocr_explain mode, <correction> holds the English translation, so only check <reply_nl> as Dutch
        if mode == "ocr_explain" or correction.upper() == "NONE":
            dutch_blocks = [reply_nl]
        else:
            dutch_blocks = [reply_nl, correction]

        for text in dutch_blocks:
            tokens = [t.lower() for t in TOKEN_RE.findall(text)]
            if len(tokens) < 2:
                errors.append(f"Row {idx} ({sid}): Dutch text too short")
                failed_ids.append(sid)
                break
            if (sum(1 for t in tokens if t in valid_words) / len(tokens)) < 0.90:
                unknown = [t for t in tokens if t not in valid_words]
                errors.append(f"Row {idx} ({sid}): unknown Dutch words {unknown[:4]}")
                failed_ids.append(sid)
                break

            gender_err = None
            for art, noun in ARTICLE_NOUN_RE.findall(text):
                art_l, noun_l = art.lower(), noun.lower()
                if noun_l.endswith("je") and art_l != "het":
                    gender_err = f"diminutive '{art_l} {noun_l}' must be 'het'"
                    break
                if noun_l in noun_articles:
                    exp = noun_articles[noun_l]
                    if exp in ("de", "het") and art_l != exp:
                        gender_err = f"'{art_l} {noun_l}' should be '{exp} {noun_l}'"
                        break
            if gender_err:
                errors.append(f"Row {idx} ({sid}): {gender_err}")
                failed_ids.append(sid)
                break

    if errors:
        print(f"FAIL: {len(errors)}/{len(seeds)} rows failed:")
        for err in errors[:30]:
            print("  -", err)
        print("\nFailed IDs:", ", ".join(failed_ids))
        return 1

    print(f"PASS: {len(seeds)}/{len(seeds)} rows (seeds {args.start}..{end_idx}) verified clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
