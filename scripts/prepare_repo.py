#!/usr/bin/env python3
"""
Run ONCE locally before pushing your repo to GitHub:
  python scripts/prepare_repo.py --batches 10 --batch-size 250
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

CHAT_SCENARIOS = [
    "NS train delay, platform change, or OV-chipkaart issue at a Dutch station",
    "Buying, negotiating, or picking up a secondhand item via Marktplaats",
    "Ordering coffee, broodjes, or borrel snacks at a café",
    "Making an appointment or explaining symptoms at the huisarts",
    "Daily standup or quick Slack/coffee chat with Dutch coworkers",
    "Dealing with gemeente registration, DigiD, or housing/energy bills",
    "Small talk about Dutch weather, cycling in headwind, or weekend plans",
    "Supermarket checkout (Albert Heijn / Jumbo) and Bonuskaart questions",
    "Rescheduling plans with a Dutch friend over WhatsApp",
    "Asking for directions or bicycle repair help at a fietsenmaker",
]

OCR_SCENARIOS = [
    "Dutch street or bicycle parking prohibition sign (e.g., Geen fietsen plaatsen)",
    "NS station platform display or rail disruption notice (werkzaamheden / omleiding)",
    "Official letter snippet from gemeente, Belastingdienst, or housing corporation",
    "Albert Heijn / Jumbo supermarket shelf label, discount rule, or food packaging",
    "Huisarts clinic door notice, pharmacy label, or prescription dosage instruction",
    "Café menu board, terrace notice, or Dutch receipt line items",
    "Apartment building hallway notice (afval, grofvuil, huisregels)",
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

# 55% learner_error, 15% valid_dutch, 15% how_do_i_say, 15% ocr_explain
INPUT_MODES = (
    ["learner_error"] * 11
    + ["valid_dutch"] * 3
    + ["how_do_i_say"] * 3
    + ["ocr_explain"] * 3
)

# High-frequency everyday Dutch nouns for realistic [DB_HINTS]
EVERYDAY_NOUNS = [
    "trein", "fiets", "station", "kaartje", "perron", "vertraging", "bus", "auto",
    "huis", "kamer", "keuken", "raam", "deur", "sleutel", "tafel", "stoel", "bed",
    "werk", "kantoor", "vergadering", "verslag", "afspraak", "collega", "baan",
    "winkel", "bon", "tas", "brood", "koffie", "water", "bier", "rekening", "prijs",
    "arts", "ziekenhuis", "recept", "pijn", "hoofd", "keel", "medicijn",
    "weer", "regen", "wind", "zon", "paraplu", "jas", "schoen", "weekend", "vakantie",
    "bericht", "telefoon", "oplader", "scherm", "batterij", "gemeente", "formulier",
    "bord", "stoep", "gevel", "afval", "brief", "adres", "korting", "kassa", "balie",
]

# Curated everyday + separable verbs so seed hints stay usable at A1/A2 level
EVERYDAY_VERBS = [
    # Core everyday verbs
    "reizen", "fietsen", "lopen", "wachten", "vertrekken", "betalen", "bestellen",
    "kopen", "verkopen", "huren", "werken", "sturen", "bellen", "vragen", "antwoorden",
    "zoeken", "vinden", "vergeten", "onthouden", "begrijpen", "uitleggen", "helpen",
    "maken", "repareren", "testen", "checken", "regelen", "boeken", "annuleren",
    "eten", "drinken", "koken", "slapen", "sporten", "wandelen", "zwemmen",
    # Common separable verbs (scheidbare werkwoorden — high value for grammar training)
    "opbellen", "meenemen", "afspreken", "doorgeven", "uitstappen", "instappen",
    "overstappen", "aankomen", "ophalen", "terugbrengen", "invullen", "aanvragen",
    "afrekenen", "uitzoeken", "nakijken", "opladen", "aanzetten", "uitzetten",
    "opruimen", "schoonmaken", "langskomen", "terugbellen", "doorsturen", "afzeggen",
]

# Wiktionary's separable tag is unreliable for these lemmas (flag often 0), so fall
# back to the stressed-prefix test. "aanvragen" is deliberately absent: it also
# exists as a non-separable noun-derived verb, so we trust the DB flag for it.
SEPARABLE_PREFIXES = (
    "op", "af", "aan", "uit", "in", "over", "na", "toe", "terug", "door",
    "mee", "langs", "bij", "weg", "vast", "samen", "thuis",
)


def is_separable_verb(verb: str, db_flag: int = 0) -> bool:
    if db_flag:
        return True
    return any(verb.startswith(p) and len(verb) - len(p) >= 3 for p in SEPARABLE_PREFIXES)


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

    print("Streaming Dutch Wiktionary (Kaikki) for de/het genders & infinitive verbs...")
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

            tags = set(entry.get("tags", []))
            for sense in entry.get("senses", []):
                tags.update(sense.get("tags", []))

            # Skip inflected forms (participles, past tenses, plurals) for lemma tables
            if "form-of" in tags or "alt-of" in tags:
                continue

            if pos == "noun":
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
          f"{len(words):,} words, {len(noun_rows):,} de/het nouns, {len(verb_rows):,} verb lemmas.")


def generate_seed_batches(num_batches: int, batch_size: int) -> None:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / ".gitkeep").touch()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    noun_map = dict(
        cur.execute("SELECT lemma, article FROM nouns WHERE article IN ('de', 'het')").fetchall()
    )
    verb_map = dict(cur.execute("SELECT lemma, separable FROM verbs").fetchall())
    conn.close()

    common_nouns = [(w, noun_map[w]) for w in EVERYDAY_NOUNS if w in noun_map]
    common_verbs = [(v, verb_map[v]) for v in EVERYDAY_VERBS if v in verb_map]

    missing_nouns = [w for w in EVERYDAY_NOUNS if w not in noun_map]
    missing_verbs = [v for v in EVERYDAY_VERBS if v not in verb_map]
    if missing_nouns or missing_verbs:
        print(f"WARN: not in lexicon — nouns {missing_nouns}, verbs {missing_verbs}")

    rng = random.Random(42)
    # Start at 2 so seeds/batch_001.jsonl and data/batch_001.jsonl stay 1-to-1 aligned
    for b in range(2, num_batches + 1):
        batch_file = SEEDS_DIR / f"batch_{b:03d}.jsonl"
        with open(batch_file, "w", encoding="utf-8") as f:
            for idx in range(1, batch_size + 1):
                mode = rng.choice(INPUT_MODES)
                n1, n2 = rng.sample(common_nouns, 2)
                v1 = rng.choice(common_verbs)
                hints = [
                    f"{n1[0]}={n1[1]}",
                    f"{n2[0]}={n2[1]}",
                    f"{v1[0]}={'separable_verb' if is_separable_verb(v1[0], v1[1]) else 'verb'}",
                ]
                scenario = rng.choice(OCR_SCENARIOS if mode == "ocr_explain" else CHAT_SCENARIOS)
                seed = {
                    "id": f"b{b:03d}_{idx:03d}",
                    "cefr": rng.choice(CEFR_LEVELS),
                    "register": rng.choice(REGISTERS),
                    "scenario": scenario,
                    "trap": rng.choice(GRAMMAR_TRAPS),
                    "mode": mode,
                    "db_hints": ", ".join(hints),
                    "anchor_words": [n1[0], n2[0], v1[0]],
                }
                f.write(json.dumps(seed, ensure_ascii=False) + "\n")
    print(f"Wrote seed batches 002..{num_batches:03d} ({batch_size} seeds each) to {SEEDS_DIR}/; "
          f"batch_001 left untouched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    build_lexicon_db()
    generate_seed_batches(args.batches, args.batch_size)
