#!/usr/bin/env python3
"""Fast diagnostic preflight mirroring validate_batch.py, with verbose detail.

Prints every row that would fail and why (unknown-word ratios, article
conflicts, length, mode rules), plus anchor-word and English-leak hints.
Read-only: never edits the batch.
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

DB_PATH = Path("assets/dutch_lexicon.db")
TOKEN_RE = re.compile(r"[a-zàáâäèéêëìíîïòóôöùúûüĳ'-]+", re.IGNORECASE)
ARTICLE_NOUN_RE = re.compile(r"\b(de|het)\s+([a-zàáâäèéêëìíîïòóôöùúûüĳ'-]{3,})\b", re.IGNORECASE)
ENGLISH_HINTS = {
    "the", "and", "you", "your", "with", "please", "sorry", "yes", "no", "is", "are",
    "not", "have", "this", "that", "what", "how", "where", "thanks", "thank", "hello",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_jsonl", type=Path)
    ap.add_argument("output_jsonl", type=Path)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-words", type=int, default=105)
    ap.add_argument("--max-chars", type=int, default=700)
    args = ap.parse_args()

    seeds = [json.loads(l) for l in args.seed_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    end = args.end or len(seeds)
    seeds = seeds[args.start - 1:end]
    out = [l for l in args.output_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

    conn = sqlite3.connect(DB_PATH)
    words = {r[0] for r in conn.execute("SELECT word FROM words")}
    nouns = {r[0]: r[1] for r in conn.execute("SELECT lemma, article FROM nouns")}
    conn.close()

    problems = 0
    if len(out) != len(seeds):
        print(f"COUNT: expected {len(seeds)}, got {len(out)}")
        problems += 1

    tag_re = re.compile(
        r"^<reply_nl>(?P<r>.*?)</reply_nl>\s*<correction>(?P<c>.*?)</correction>\s*"
        r"<rule_en>(?P<ru>.*?)</rule_en>\s*<vocab>(?P<v>.*?)</vocab>$", re.DOTALL
    )

    for i, (seed, line) in enumerate(zip(seeds, out), start=args.start):
        issues = []
        row = json.loads(line)
        msgs = row["messages"]
        a = msgs[2]["content"]
        m = tag_re.match(a)
        if row["id"] != seed["id"]:
            issues.append(f"id mismatch {row['id']} != {seed['id']}")
        if not m:
            print(f"{seed['id']}: MALFORMED TAGS")
            problems += 1
            continue
        w, c = len(a.split()), len(a)
        if w > args.max_words or c > args.max_chars:
            issues.append(f"length {w}w/{c}c")
        if a.count("<reply_nl>") != 1 or a.count("</vocab>") != 1:
            issues.append("tag count")
        r, corr = m.group("r").strip(), m.group("c").strip()
        if "</" in r or "<" in r:
            issues.append("stray tag in reply_nl")
        if seed["mode"] == "valid_dutch" and corr.upper() != "NONE":
            issues.append("valid_dutch needs NONE")
        if seed["mode"] == "learner_error" and corr.upper() == "NONE":
            issues.append("learner_error needs correction")
        if seed["mode"] == "ocr_explain":
            if not msgs[1]["content"].startswith("[OCR_TEXT]"):
                issues.append("ocr needs [OCR_TEXT]")
            if corr.upper() == "NONE" or len(corr.split()) < 2:
                issues.append("ocr needs translation")

        blocks = [r] if (seed["mode"] == "ocr_explain" or corr.upper() == "NONE") else [r, corr]
        for text in blocks:
            toks = [t.lower() for t in TOKEN_RE.findall(text)]
            unknown = [t for t in toks if t not in words]
            ratio = 1 - len(unknown) / max(len(toks), 1)
            if ratio < 0.90:
                issues.append(f"lexicon {ratio:.0%} unknown={unknown[:6]}")
            for art, noun in ARTICLE_NOUN_RE.findall(text):
                al, nl = art.lower(), noun.lower()
                if nl.endswith("je") and al != "het":
                    issues.append(f"diminutive '{al} {nl}'")
                elif nl in nouns and nouns[nl] in ("de", "het") and al != nouns[nl]:
                    issues.append(f"article '{al} {nl}' -> '{nouns[nl]} {nl}'")

        # style hints (non-fatal)
        hint = ""
        en = [t for t in TOKEN_RE.findall(r.lower()) if t in ENGLISH_HINTS]
        if en:
            hint += f" ENGLISH?{sorted(set(en))}"
        anchors = [a0 for a0 in seed["anchor_words"] if re.search(rf"\b{re.escape(a0.lower())}", (msgs[1]["content"] + " " + m.group("r")).lower())]
        if not anchors:
            hint += " no-anchor"
        if issues:
            problems += 1
            print(f"{seed['id']} [{seed['mode']}]: " + "; ".join(issues) + hint)
        elif hint.strip():
            print(f"{seed['id']} ok{hint}")

    print(f"\n{'CLEAN' if problems == 0 else f'{problems} problem rows'} over seeds {args.start}..{end}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
