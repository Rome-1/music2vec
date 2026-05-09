"""01b_label.py — derive per-taxonomy CSVs from works.csv title patterns.

Rule-based labeler for the Bach-heavy pilot corpus. For each taxonomy
defined in music2vec.taxonomies, emit data/taxonomies/<taxonomy>.csv
with columns (work_id, label) for the works the rules match.

Conservative on purpose: a work without a confident match gets no label
in that taxonomy (rather than being mislabeled). Hand-edits to the
emitted CSVs are welcome — this script writes only when the file is
absent unless --overwrite is passed.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKS_CSV = ROOT / "data" / "works.csv"
TAX_DIR = ROOT / "data" / "taxonomies"


def label_compositional_device(row: dict) -> str | None:
    title = row["title"].lower()
    wid = row["work_id"].lower()
    # Fugues — Bach is generous with the label (Fuga, Fugue, fuga, fugue),
    # plus the suffix in his solo-violin movements (BWV 1001 Fuga, etc.)
    if re.search(r"\bfug(a|ue|hetta|ato)\b", title):
        return "fugue"
    if "fuga" in wid or "fugue" in wid:
        return "fugue"
    # Canons
    if re.search(r"\bcanon\b", title) or "canon" in wid:
        return "canon"
    # Passacaglia / chaconne / ground bass
    if re.search(r"passacaglia|chaconne|ciacona", title):
        return "passacaglia"
    # Theme & variations
    if re.search(r"variation|aria.*variata|goldberg", title):
        return "theme_variations"
    return None


def label_dance_type(row: dict) -> str | None:
    title = row["title"].lower()
    wid = row["work_id"].lower()
    parent = (row.get("parent_work_id") or "").lower()
    txt = f"{title} {wid} {parent}"

    patterns = [
        ("allemande", r"\ballemand[ae]\b"),
        ("courante",  r"\b(courante|corrente)\b"),
        ("sarabande", r"\bsaraband[ae]\b"),
        ("gigue",     r"\b(gigu?e|gigaa?|jig)\b"),
        ("gavotte",   r"\bgavotte?\b"),
        ("bourree",   r"\bbour?r[eé]e\b"),
        ("menuet",    r"\b(men[uu]et+o?|minuet)\b"),
        ("siciliana", r"\bsiciliana?o?\b"),
        ("tarantella", r"\btarantella\b"),
        ("waltz",     r"\b(walzer|waltz|valse)\b"),
        ("mazurka",   r"\bmazurk[ae]\b"),
        ("polonaise", r"\b(polonaise|polonez)\b"),
        ("habanera",  r"\bhabanera\b"),
        ("march",     r"\b(marche?|marsch)\b"),
    ]
    label_remap = {"menuet": "minuet"}
    for label, pat in patterns:
        if re.search(pat, txt):
            return label_remap.get(label, label)
    return None


def label_instrumentation(row: dict) -> str | None:
    instr = (row.get("instrumentation_hint", "") or "").lower()
    parent = (row.get("parent_work_id") or "").lower()
    title = (row.get("title") or "").lower()
    wid = (row.get("work_id") or "").lower()

    # Mutopia's mutopiainstrument is usually authoritative.
    if "harpsichord" in instr and "piano" in instr:
        return "solo_keyboard_harpsichord"  # WTC marked "Harpsichord, Piano"
    if "harpsichord" in instr:
        return "solo_keyboard_harpsichord"
    if "piano" in instr and "voice" not in instr:
        return "solo_keyboard_piano"
    if "organ" in instr:
        return "solo_keyboard_organ"
    if "guitar" in instr or "lute" in instr:
        return "lute_guitar_solo"
    if "string quartet" in instr or "quatuor à cordes" in instr:
        return "string_quartet"
    if any(k in instr for k in ("orchestra", "orchestre", "symphony")):
        return "full_orchestra"
    if "choir" in instr or "chorus" in instr or "satb" in instr:
        return "satb_a_cappella"
    if "violin" in instr or "violine" in instr:
        return "unaccompanied_string"
    if "cello" in instr or "violoncello" in instr:
        return "unaccompanied_string"
    if "viola" in instr:
        return "unaccompanied_string"

    # Fall back to BWV / opus / composer-specific heuristics
    bwv = re.search(r"bwv(\d+)", parent)
    if bwv:
        n = int(bwv.group(1))
        if 846 <= n <= 893:        # WTC books I + II
            return "solo_keyboard_harpsichord"
        if 1001 <= n <= 1006:      # solo violin sonatas + partitas
            return "unaccompanied_string"
        if 1007 <= n <= 1012:      # cello suites
            return "unaccompanied_string"

    # Beethoven sonatas / Chopin works are piano unless said otherwise
    if "beethovenlv" in parent and "sonate" in title.replace(" ", ""):
        return "solo_keyboard_piano"
    if "chopinff" in parent:
        return "solo_keyboard_piano"
    if "moonlight" in wid or "pathetique" in wid:
        return "solo_keyboard_piano"
    if "dvoraka" in parent and ("quartet" in title.lower()
                                 or "americanquartet" in wid):
        return "string_quartet"
    if "vivaldia" in parent:
        return "full_orchestra"  # Vivaldi concerti grossi / four seasons
    if "mozartwa" in parent:
        if "kv525" in parent or "nachtmusik" in title.lower():
            return "string_quartet"
        if "kv550" in parent or "kv551" in parent or "symphony" in title.lower():
            return "full_orchestra"
        if "requiem" in title.lower() or "kv626" in parent:
            return "mass_with_orchestra"
        if "averum" in wid or "kv618" in parent:
            return "satb_a_cappella"
    return None


def label_opus_cycle(row: dict) -> str | None:
    parent = (row.get("parent_work_id") or "").lower()
    title = (row.get("title") or "").lower()
    bwv = re.search(r"bwv(\d+)", parent)
    if bwv:
        n = int(bwv.group(1))
        if 846 <= n <= 869:
            return "bach_wtc_1"
        if 870 <= n <= 893:
            return "bach_wtc_2"
        if 1001 <= n <= 1006:
            return "bach_solo_violin"
        if 1007 <= n <= 1012:
            return "bach_cello_suites"
    if "wohltemperierte clavier i," in title or "wtc i" in title:
        return "bach_wtc_1"
    if "wohltemperierte clavier ii" in title or "wtc ii" in title:
        return "bach_wtc_2"
    if "goldberg" in title:
        return "bach_goldberg"
    if "art of fugue" in title or "kunst der fuge" in title:
        return "bach_art_of_fugue"

    # Chopin opus cycles
    if "chopinff-o28" in parent:
        return "chopin_op28_preludes"
    if "chopinff-o10" in parent:
        return "chopin_op10_etudes"
    if "chopinff-o25" in parent:
        return "chopin_op25_etudes"

    # Beethoven 32 sonatas — anything labeled lvb-sonate-NNN
    if "beethovenlv" in parent and (
        "sonate" in title or "moonlight" in title or "pathetique" in title
        or re.search(r"sonata", title)
    ):
        return "beethoven_32_sonatas"
    return None


def label_national_school(row: dict) -> str | None:
    composer = (row.get("composer") or "").lower()
    parent = (row.get("parent_work_id") or "").lower()

    if composer in {"bachjs", "bachj", "bach"} or "bachjs" in parent:
        return "german_contrapuntal"
    if "beethovenlv" in parent or composer == "beethovenlv":
        return "german_contrapuntal"
    if "mozartwa" in parent or composer == "mozartwa":
        return "german_contrapuntal"  # Austrian; Viennese Classical
    if "handelgf" in parent or composer == "handelgf":
        return "german_contrapuntal"  # German-born; later naturalized British
    if "chopinff" in parent or composer == "chopinff":
        return "polish_romantic"
    if "dvoraka" in parent or composer == "dvoraka":
        return "czech_nationalist"
    if "vivaldia" in parent or composer == "vivaldia":
        return "italian_operatic"
    return None


def label_sacred_function(row: dict) -> str | None:
    title = (row.get("title") or "").lower()
    parent = (row.get("parent_work_id") or "").lower()
    wid = (row.get("work_id") or "").lower()
    txt = f"{title} {parent} {wid}"

    if "spiritus domini" in title:
        return "motet"
    if "ave verum" in txt or "ave, verum" in txt or "kv618" in parent or "aveverum" in wid or "verumm" in wid:
        return "motet"
    if "magnificat" in txt:
        return "magnificat"
    if "stabat mater" in txt:
        return "stabat_mater"
    if "requiem" in txt or "kv626" in parent or "dies-irae" in wid:
        return "requiem"
    if "passion" in title or "passio" in title:
        return "passion"
    if "te deum" in txt:
        return "te_deum"
    if "messiah" in txt or "hwv56" in parent:
        return "oratorio"
    if "chorale" in title and "prelude" in title:
        return "chorale_prelude"
    if "chorale" in title:
        return "lutheran_chorale"
    return None


LABELERS = {
    "compositional_device": label_compositional_device,
    "dance_type":           label_dance_type,
    "instrumentation":      label_instrumentation,
    "opus_cycle":           label_opus_cycle,
    "national_school":      label_national_school,
    "sacred_function":      label_sacred_function,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing taxonomy CSVs")
    args = ap.parse_args()

    if not WORKS_CSV.exists():
        print(f"missing {WORKS_CSV}; run 01_acquire.py first")
        return 1

    rows = list(csv.DictReader(WORKS_CSV.open()))
    TAX_DIR.mkdir(parents=True, exist_ok=True)

    for tax_name, fn in LABELERS.items():
        out_path = TAX_DIR / f"{tax_name}.csv"
        if out_path.exists() and not args.overwrite:
            print(f"  skip (exists): {out_path}")
            continue
        n = 0
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["work_id", "label"])
            for row in rows:
                lbl = fn(row)
                if lbl:
                    w.writerow([row["work_id"], lbl])
                    n += 1
        print(f"  wrote {n} labels -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
