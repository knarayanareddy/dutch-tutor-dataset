#!/usr/bin/env python3
"""
Deterministic validation gate executed by Sonnet 5.5 inside Arena.ai Agent Mode.
Usage:
  python scripts/validate_batch.py seeds/batch_001.jsonl data/batch_001.jsonl
"""

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
    if len(sys.argv) != 3:
        print("Usage: python scripts/validate_batch.py <seed_jsonl> <output_jsonl>")
        return 2

    seed_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    if not out_path.exists():
        print(f"FAIL: Output file {out_path} does not exist.")
        return 1

    seeds = [json.loads(line) for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if len(seeds) != len(out_lines):
        print(f"FAIL: Line count mismatch — expected {len(seeds)} rows, got {len(out_lines)}.")
        return 1

    conn = sqlite3.connect(DB_PATH)
    valid_words = {r[0] for r in conn.execute("SELECT word FROM words")}
    noun_articles = {r[0]: r[1] for r in conn.execute("SELECT lemma, article FROM nouns")}
    conn.close()

    errors = []
    for idx, (seed, raw_line) in enumerate(zip(seeds, out_lines), start=1):
        try:
            row = json.loads(raw_line)
            msgs = row["messages"]
            assert len(msgs) == 3
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user" and msgs[1]["content"].strip()
            assert msgs[2]["role"] == "assistant"
        except Exception as e:
            errors.append(f"Line {idx} ({seed['id']}): invalid JSON/messages structure ({e})")
            continue

        expected_sys = f"Je bent een Nederlandse taalcoach. Level: {seed['cefr']}. [DB_HINTS: {seed['db_hints']}]"
        if msgs[0]["content"].strip() != expected_sys:
            errors.append(f"Line {idx} ({seed['id']}): system prompt must equal `{expected_sys}`")
            continue

        asst = msgs[2]["content"].strip()
        m = TAG_PATTERN.match(asst)
        if not m:
            errors.append(f"Line {idx} ({seed['id']}): missing or out-of-order XML tags")
            continue

        if len(asst.split()) > 115 or len(asst) > 720:
            errors.append(f"Line {idx} ({seed['id']}): exceeds 115 words / 720 chars ({len(asst.split())} words)")
            continue

        reply_nl = m.group("reply_nl").strip()
        correction = m.group("correction").strip()
        rule_en = m.group("rule_en").strip()
        vocab = m.group("vocab").strip()

        if not reply_nl or not rule_en or not vocab:
            errors.append(f"Line {idx} ({seed['id']}): empty XML tag")
            continue

        mode = seed["mode"]
        if mode == "valid_dutch" and correction.upper() != "NONE":
            errors.append(f"Line {idx} ({seed['id']}): mode=valid_dutch requires <correction>NONE</correction>")
            continue
        if mode == "learner_error" and correction.upper() == "NONE":
            errors.append(f"Line {idx} ({seed['id']}): mode=learner_error cannot have <correction>NONE</correction>")
            continue

        dutch_blocks = [reply_nl] + ([] if correction.upper() == "NONE" else [correction])
        for text in dutch_blocks:
            tokens = [t.lower() for t in TOKEN_RE.findall(text)]
            if len(tokens) < 2:
                errors.append(f"Line {idx} ({seed['id']}): Dutch text too short")
                break
            known_ratio = sum(1 for t in tokens if t in valid_words) / len(tokens)
            if known_ratio < 0.90:
                unknown = [t for t in tokens if t not in valid_words]
                errors.append(f"Line {idx} ({seed['id']}): unknown/non-Dutch words {unknown[:4]}")
                break

            gender_err = None
            for art, noun in ARTICLE_NOUN_RE.findall(text):
                art_l, noun_l = art.lower(), noun.lower()
                if noun_l.endswith("je") and art_l != "het":
                    gender_err = f"diminutive '{art_l} {noun_l}' must use 'het'"
                    break
                if noun_l in noun_articles:
                    exp = noun_articles[noun_l]
                    if exp in ("de", "het") and art_l != exp:
                        gender_err = f"'{art_l} {noun_l}' should be '{exp} {noun_l}'"
                        break
            if gender_err:
                errors.append(f"Line {idx} ({seed['id']}): gender mismatch ({gender_err})")
                break

    if errors:
        print(f"FAIL: {len(errors)}/{len(seeds)} rows failed validation:")
        for err in errors[:25]:
            print("  -", err)
        return 1

    print(f"PASS: {len(seeds)}/{len(seeds)} rows verified clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
