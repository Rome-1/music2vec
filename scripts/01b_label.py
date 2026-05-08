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
    txt = f"{title} {wid}"

    patterns = [
        ("allemande", r"\ballemand[ae]\b"),
        ("courante",  r"\bcourante|corrente\b"),
        ("sarabande", r"\bsaraband[ae]\b"),
        ("gigue",     r"\bgigu?e\b|\bgigaa?\b|\bjig\b"),
        ("gavotte",   r"\bgavotte?\b"),
        ("bourree",   r"\bbour?r[eé]e\b"),
        ("menuet",    r"\bmen[uu]et+o?\b|\bminuet\b"),
        ("siciliana", r"\bsiciliana?o?\b"),
        ("tarantella", r"\btarantella\b"),
        ("waltz",     r"\bwalzer|\bwaltz|\bvalse\b"),
        ("mazurka",   r"\bmazurk[ae]\b"),
        ("polonaise", r"\bpolonaise|polonez\b"),
        ("habanera",  r"\bhabanera\b"),
        ("march",     r"\bmarche?|marsch\b"),
    ]
    # Map "menuet" to "minuet"-equivalent label in our taxonomy: keep
    # as separate label since taxonomies.py uses "minuet" — adjust:
    label_remap = {"menuet": "minuet"}
    for label, pat in patterns:
        if re.search(pat, txt):
            return label_remap.get(label, label)
    return None


def label_instrumentation(row: dict) -> str | None:
    instr = (row.get("instrumentation_hint", "") or "").lower()
    parent = row.get("parent_work_id", "")

    # Mutopia's mutopiainstrument is usually authoritative.
    if "harpsichord" in instr and "piano" in instr:
        return "solo_keyboard_harpsichord"  # WTC marked "Harpsichord, Piano"
    if "harpsichord" in instr:
        return "solo_keyboard_harpsichord"
    if "piano" in instr and "voice" not in instr:
        return "solo_keyboard_piano"
    if "organ" in instr:
        return "solo_keyboard_organ"
    if "violin" in instr or "violine" in instr:
        return "unaccompanied_string"
    if "cello" in instr or "violoncello" in instr:
        return "unaccompanied_string"
    if "viola" in instr:
        return "unaccompanied_string"

    # Fall back to parent-work BWV ranges for Bach
    bwv = re.search(r"bwv(\d+)", parent or "")
    if bwv:
        n = int(bwv.group(1))
        if 846 <= n <= 893:        # WTC books I + II
            return "solo_keyboard_harpsichord"
        if 1001 <= n <= 1006:      # solo violin sonatas + partitas
            return "unaccompanied_string"
        if 1007 <= n <= 1012:      # cello suites
            return "unaccompanied_string"
    return None


def label_opus_cycle(row: dict) -> str | None:
    parent = row.get("parent_work_id", "") or ""
    title = row["title"].lower()
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
    return None


def label_national_school(row: dict) -> str | None:
    composer = (row.get("composer") or "").lower()
    parent = (row.get("parent_work_id") or "").lower()
    if composer in {"bachjs", "bachj", "bach"} or "bachjs" in parent:
        return "german_contrapuntal"
    return None


def label_sacred_function(row: dict) -> str | None:
    title = row["title"].lower()
    if "spiritus domini" in title:
        return "motet"
    if "magnificat" in title:
        return "magnificat"
    if "stabat mater" in title:
        return "stabat_mater"
    if "requiem" in title:
        return "requiem"
    if "passion" in title or "passio" in title:
        return "passion"
    if "te deum" in title:
        return "te_deum"
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
