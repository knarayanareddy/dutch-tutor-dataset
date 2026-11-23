#!/usr/bin/env python3
"""
Run ONCE locally before pushing your repo to GitHub:
  python scripts/prepare_repo.py --batches 25 --batch-size 100
"""

import argparse
import gzip
import json
import random
import re
import sqlite3
import urllib.request
from pathlib import Path

DB_PATH = Path("assets/dutch_lexicon.db")
SEEDS_DIR = Path("seeds")
DATA_DIR = Path("data")

OPENTAAL_URL = "https://raw.githubusercontent.com/OpenTaal/opentaal-wordlist/master/wordlist.txt"
KAIKKI_NL_URL = "https://kaikki.org/dictionary/downloads/nl/nl-extract.jsonl.gz"

CEFR_LEVELS = ["A1", "A2", "A2", "B1", "B1", "B2"]

REGISTERS = [
    "casual_spoken (natural spoken Dutch: je/jij, even/effe, hoor, toch, gewoon)",
    "neutral_everyday (standard polite-informal Dutch with colleagues or neighbors)",
    "polite_formal (use u, clear polite phrasing for gemeente, huisarts, or work emails)",
]

SCENARIOS = [
    "NS train delay, platform change, or OV-chipkaart issue at a Dutch station",
    "Buying, negotiating, or picking up a secondhand item via Marktplaats",
    "Ordering coffee, broodjes, or borrel snacks at a café",
    "Making an appointment or describing symptoms at the huisarts",
    "Daily standup or quick Slack/coffee chat with Dutch coworkers",
    "Dealing with gemeente registration, DigiD, or housing/energy bills",
    "Small talk about Dutch weather, cycling in headwind, or weekend plans",
    "Supermarket checkout (Albert Heijn / Jumbo) and Bonuskaart questions",
    "Rescheduling plans with a Dutch friend over WhatsApp",
    "Asking for directions or bicycle repair help at a fietsenmaker",
]

GRAMMAR_TRAPS = [
    "de vs. het definite article agreement (and het-noun adjective -e inflection rule)",
    "V2 main-clause word order inversion after fronting time/place (e.g., 'Morgen ga ik...')",
    "subordinate clause verb-final word order after omdat, dat, als, wanneer, terwijl",
    "separable verbs (scheidbare werkwoorden) splitting in main clauses vs. staying together in te-infinitives",
    "usage of 'er' (locative, quantitative 'er twee', or prepositional 'eraan/ermee')",
    "negation with 'geen' (indefinite nouns) vs. 'niet' (verbs, adjectives, definite nouns)",
    "perfectum auxiliary choice ('hebben' vs. 'zijn' for motion/change-of-state verbs)",
    "diminutives (-je, -tje, -pje, -etje) which always take 'het'",
    "modal verbs + infinitive placement at the end of the clause",
    "false friends from English (e.g., 'eventueel', 'actueel', 'brutaal', 'slim', 'winkel')",
]

# 65% flawed learner Dutch, 20% valid Dutch (prevents over-correction), 15% how-to-say query
INPUT_MODES = ["learner_error"] * 13 + ["valid_dutch"] * 4 + ["how_do_i_say"] * 3


def build_lexicon_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE words (word TEXT PRIMARY KEY);
        CREATE TABLE nouns (lemma TEXT PRIMARY KEY, article TEXT NOT NULL, plural TEXT);
        CREATE TABLE verbs (lemma TEXT PRIMARY KEY, separable INTEGER DEFAULT 0);
    """)

    print("Downloading OpenTaal wordlist...")
    words = set()
    req = urllib.request.Request(
        OPENTAAL_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    for raw_line in urllib.request.urlopen(req):
        w = raw_line.decode("utf-8", errors="ignore").strip().lower()
        if w and re.match(r"^[a-zàáâäèéêëìíîïòóôöùúûüĳ'-]+$", w):
            words.add(w)

    print("Streaming Dutch Wiktionary (Kaikki) for de/het genders & verbs...")
    noun_rows = {}
    verb_rows = {}
    req_kaikki = urllib.request.Request(
        KAIKKI_NL_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with gzip.GzipFile(fileobj=urllib.request.urlopen(req_kaikki)) as gz:
        for line in gz:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("lang_code") != "nl":
                continue

            word = (entry.get("word") or "").strip().lower()
            pos = entry.get("pos")
            if not word or " " in word:
                continue
            words.add(word)

            for form_obj in entry.get("forms", []):
                f = (form_obj.get("form") or "").strip().lower()
                if f and " " not in f:
                    words.add(f)

            if pos == "noun":
                tags = set(entry.get("tags", []))
                for sense in entry.get("senses", []):
                    tags.update(sense.get("tags", []))
                is_neuter = "neuter" in tags
                is_common = bool(tags & {"common", "masculine", "feminine"})
                for ht in entry.get("head_templates", []):
                    exp = (ht.get("expansion") or "").lower()
                    if exp.startswith("het "):
                        is_neuter = True
                    elif exp.startswith("de "):
                        is_common = True

                if is_neuter and is_common:
                    article = "de/het"
                elif is_neuter:
                    article = "het"
                elif is_common:
                    article = "de"
                else:
                    continue

                plural = next(
                    (fo.get("form", "").strip().lower()
                     for fo in entry.get("forms", [])
                     if "plural" in fo.get("tags", [])),
                    None,
                )
                noun_rows[word] = (word, article, plural)

            elif pos == "verb":
                separable = int(
                    any("separable" in fo.get("tags", []) for fo in entry.get("forms", []))
                )
                verb_rows[word] = (word, separable)

    cur.executemany("INSERT OR IGNORE INTO words(word) VALUES (?)", [(w,) for w in words])
    cur.executemany("INSERT OR REPLACE INTO nouns VALUES (?, ?, ?)", noun_rows.values())
    cur.executemany("INSERT OR REPLACE INTO verbs VALUES (?, ?)", verb_rows.values())
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    print(f"Built {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB): "
          f"{len(words):,} words, {len(noun_rows):,} de/het nouns, {len(verb_rows):,} verbs.")


def generate_seed_batches(num_batches: int, batch_size: int) -> None:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / ".gitkeep").touch()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    nouns = cur.execute(
        "SELECT lemma, article FROM nouns WHERE length(lemma) BETWEEN 4 AND 12 "
        "AND article IN ('de', 'het')"
    ).fetchall()
    verbs = cur.execute(
        "SELECT lemma, separable FROM verbs WHERE length(lemma) BETWEEN 5 AND 13"
    ).fetchall()
    conn.close()

    rng = random.Random(42)
    for b in range(1, num_batches + 1):
        batch_file = SEEDS_DIR / f"batch_{b:03d}.jsonl"
        with open(batch_file, "w", encoding="utf-8") as f:
            for idx in range(1, batch_size + 1):
                n1, n2 = rng.sample(nouns, 2)
                v1 = rng.choice(verbs)
                hints = [
                    f"{n1[0]}={n1[1]}",
                    f"{n2[0]}={n2[1]}",
                    f"{v1[0]}={'separable_verb' if v1[1] else 'verb'}",
                ]
                seed = {
                    "id": f"b{b:03d}_{idx:03d}",
                    "cefr": rng.choice(CEFR_LEVELS),
                    "register": rng.choice(REGISTERS),
                    "scenario": rng.choice(SCENARIOS),
                    "trap": rng.choice(GRAMMAR_TRAPS),
                    "mode": rng.choice(INPUT_MODES),
                    "db_hints": ", ".join(hints),
                    "anchor_words": [n1[0], n2[0], v1[0]],
                }
                f.write(json.dumps(seed, ensure_ascii=False) + "\n")
    print(f"Wrote {num_batches} seed batches ({num_batches * batch_size} seeds) to {SEEDS_DIR}/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    build_lexicon_db()
    generate_seed_batches(args.batches, args.batch_size)
