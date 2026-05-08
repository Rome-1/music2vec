"""01_acquire.py — assemble a public-domain MIDI corpus from Mutopia.

Strategy (per Rome 2026-05-08, option C):
    Mutopia (clean, fully PD, ~5,600 LilyPond works) compiled to MIDI via
    lilypond, plus curated kunstderfuge top-up where Mutopia is thin
    (deferred to a follow-up commit).

Pilot mode (default): Bach WTC + solo violin + cello suites — ~100
movements covering compositional device (fugue/prelude), dance type
(allemande/courante/sarabande/gigue/etc.), and instrumentation
(harpsichord/piano vs unaccompanied string).

Inputs:  data/raw/mutopia_src/  (Mutopia .ly source mirror, gitignored)
Outputs: data/raw/<work_id>.mid (per-piece MIDI, gitignored)
         data/works.csv         (canonical work list with metadata)

Resumable: skips works whose .mid already exists.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MUTOPIA_SRC = RAW_DIR / "mutopia_src"
WORKS_CSV = ROOT / "data" / "works.csv"

WORKS_HEADER = [
    "work_id", "composer", "title", "year", "source", "source_url",
    "license", "movement_index", "parent_work_id",
    "instrumentation_hint", "era",
]

# Regex pulls every key="value" pair out of a LilyPond \header { ... } block.
# Mutopia's .ly headers use both quoted and unquoted forms; we normalize.
HEADER_KV = re.compile(r'\s*([A-Za-z]+)\s*=\s*"((?:[^"\\]|\\.)*)"', re.M)
HEADER_BLOCK = re.compile(r"\\header\s*\{(.*?)^\}", re.M | re.S)

PILOT_PREFIXES = (
    "BachJS/BWV846", "BachJS/BWV847", "BachJS/BWV848", "BachJS/BWV849",
    "BachJS/BWV850", "BachJS/BWV851", "BachJS/BWV853", "BachJS/BWV854",
    "BachJS/BWV855", "BachJS/BWV856", "BachJS/BWV857",
    "BachJS/BWV860", "BachJS/BWV861", "BachJS/BWV862", "BachJS/BWV865",
    "BachJS/BWV869", "BachJS/BWV870", "BachJS/BWV871", "BachJS/BWV875",
    "BachJS/BWV878",
    "BachJS/BWV1001", "BachJS/BWV1002", "BachJS/BWV1003", "BachJS/BWV1004",
    "BachJS/BWV1005", "BachJS/BWV1006",
    "BachJS/BWV1007", "BachJS/BWV1008", "BachJS/BWV1009", "BachJS/BWV1010",
    "BachJS/BWV1011", "BachJS/BWV1012",
)


def parse_header(ly_text: str) -> dict[str, str]:
    """Extract \\header { key = "value" ... } as a dict."""
    m = HEADER_BLOCK.search(ly_text)
    if not m:
        return {}
    block = m.group(1)
    return {k: v for k, v in HEADER_KV.findall(block)}


def slug(*parts: str) -> str:
    """work_id from path components, lowercase + safe chars."""
    s = "-".join(parts).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:64]


def era_from_style(style: str) -> str:
    """Map Mutopia's `style` field to the era bins used in taxonomies.py."""
    s = (style or "").lower()
    if any(k in s for k in ("medieval", "renaissance")):
        return "medieval_renaissance"
    if "baroque" in s:
        return "baroque"
    if "classical" in s:
        return "classical"
    if "romantic" in s:
        return "romantic"
    if any(k in s for k in ("modern", "contemporary", "20th", "21st")):
        return "modern_contemporary"
    return ""


def year_from_date(date: str) -> str:
    """Extract first 4-digit year from a date-ish string."""
    m = re.search(r"\b(1[2-9]\d{2}|20\d{2})\b", date or "")
    return m.group(1) if m else ""


def compile_to_midi(ly_path: Path, out_dir: Path, timeout: int = 90) -> Path | None:
    """Compile a .ly file with lilypond, return path to generated .midi."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cmd = ["lilypond", "--silent",
               "-dno-point-and-click",
               "-o", str(tmp_dir / ly_path.stem),
               str(ly_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError) as e:
            print(f"  ! lilypond fail {ly_path.name}: "
                  f"{getattr(e, 'returncode', 'timeout')}", file=sys.stderr)
            return None
        midis = list(tmp_dir.glob("*.midi")) + list(tmp_dir.glob("*.mid"))
        if not midis:
            return None
        # Mutopia files often emit one MIDI; if multiple, take the largest.
        midi = max(midis, key=lambda p: p.stat().st_size)
        dest = out_dir / f"{ly_path.stem}.mid"
        shutil.copy(midi, dest)
        return dest


def discover_ly(filter_prefixes: tuple[str, ...] | None) -> list[Path]:
    """Walk MUTOPIA_SRC for .ly files, optionally filter to prefix list."""
    results = []
    for p in MUTOPIA_SRC.rglob("*.ly"):
        rel = p.relative_to(MUTOPIA_SRC).as_posix()
        if filter_prefixes and not any(rel.startswith(pre)
                                       for pre in filter_prefixes):
            continue
        results.append(p)
    return sorted(results)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="Pilot subset: Bach WTC + solo violin + cello suites")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap N pieces (smoke test)")
    ap.add_argument("--no-compile", action="store_true",
                    help="Parse headers + write CSV; skip lilypond compile")
    args = ap.parse_args()

    if not MUTOPIA_SRC.exists():
        print(f"[01] missing {MUTOPIA_SRC}; expected Mutopia source mirror "
              f"(clone github.com/MutopiaProject/MutopiaProject and copy "
              f"its `ftp/` dir into here).", file=sys.stderr)
        return 1

    prefixes = PILOT_PREFIXES if args.pilot else None
    ly_files = discover_ly(prefixes)
    if args.limit:
        ly_files = ly_files[: args.limit]

    print(f"[01] discovered {len(ly_files)} .ly files "
          f"({'pilot' if args.pilot else 'full'} subset)")

    rows: list[dict] = []
    compiled = skipped = failed = 0

    # Write CSV incrementally so the next pipeline step can start in
    # parallel on whatever has compiled so far.
    WORKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_f = WORKS_CSV.open("w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=WORKS_HEADER)
    csv_w.writeheader()
    csv_f.flush()

    for ly in ly_files:
        rel = ly.relative_to(MUTOPIA_SRC)
        composer = rel.parts[0]
        opus_dir = rel.parts[1] if len(rel.parts) > 2 else ""
        piece_dir = rel.parts[-2]
        wid = slug(composer, opus_dir, piece_dir, ly.stem)

        try:
            ly_text = ly.read_text(errors="replace")
        except OSError:
            failed += 1
            continue
        hdr = parse_header(ly_text)

        midi_dest = RAW_DIR / f"{wid}.mid"
        if not args.no_compile:
            if midi_dest.exists():
                skipped += 1
            else:
                got = compile_to_midi(ly, RAW_DIR)
                if got is None:
                    failed += 1
                    continue
                # Rename to canonical work_id
                if got.name != f"{wid}.mid":
                    shutil.move(got, midi_dest)
                compiled += 1

        row = {
            "work_id": wid,
            "composer": hdr.get("mutopiacomposer") or hdr.get("composer", ""),
            "title": (hdr.get("mutopiatitle") or hdr.get("title", "")
                      or f"{opus_dir} {piece_dir}").strip(),
            "year": year_from_date(hdr.get("date", "")),
            "source": "mutopia",
            "source_url": ("https://www.mutopiaproject.org/ftp/"
                           + rel.as_posix()),
            "license": hdr.get("copyright", "Public Domain"),
            "movement_index": "",
            "parent_work_id": slug(composer, opus_dir),
            "instrumentation_hint": hdr.get("mutopiainstrument", ""),
            "era": era_from_style(hdr.get("style", "")),
        }
        rows.append(row)
        csv_w.writerow({k: row.get(k, "") for k in WORKS_HEADER})
        csv_f.flush()

        if compiled and compiled % 10 == 0:
            print(f"  ... {compiled} compiled, {skipped} skipped, "
                  f"{failed} failed")

    csv_f.close()
    print(f"[01] wrote {len(rows)} rows -> {WORKS_CSV}")
    print(f"     compiled={compiled} skipped={skipped} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
