"""01_acquire.py — assemble a public-domain MIDI corpus.

Strategy (per Rome 2026-05-08, option C):
    Mutopia (clean, fully PD, ~2k works) + curated kunstderfuge top-up
    where Mutopia is thin (Russian school, Second Viennese, etc.).

Pilot mode (default): pull a 200-work slice covering all six taxonomies,
end-to-end smoke test before scaling to ~1500.

Output: data/raw/<work_id>.mid + data/works.csv with columns
    work_id, composer, title, year, source, source_url, license,
    movement_index, parent_work_id, instrumentation_hint, era

Resumable: re-running skips works already on disk.

NOTE: This is a scaffold. Mutopia provides a static rsync mirror at
mutopia.git.sourceforge.net plus a HTML index at mutopiaproject.org.
The full ingest implementation will iterate on the index, but for the
initial commit we ship the script signature + corpus-builder helpers
so the rest of the pipeline can be developed against synthetic data.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
WORKS_CSV = ROOT / "data" / "works.csv"

WORKS_HEADER = [
    "work_id", "composer", "title", "year", "source", "source_url",
    "license", "movement_index", "parent_work_id",
    "instrumentation_hint", "era",
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WORKS_CSV.parent.mkdir(parents=True, exist_ok=True)


def acquire_mutopia(limit: int) -> list[dict]:
    """Walk Mutopia's index, return work metadata. STUB."""
    print("[01_acquire] Mutopia ingest: not yet implemented", file=sys.stderr)
    return []


def acquire_kunstderfuge(limit: int) -> list[dict]:
    """Curated kunstderfuge top-up for Russian / Second Viennese / etc. STUB."""
    print("[01_acquire] kunstderfuge ingest: not yet implemented", file=sys.stderr)
    return []


def write_works_csv(rows: list[dict]) -> None:
    ensure_dirs()
    with WORKS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WORKS_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in WORKS_HEADER})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="Pull a 200-work pilot slice covering all taxonomies")
    ap.add_argument("--limit", type=int, default=1500,
                    help="Max works to ingest (full run)")
    args = ap.parse_args()

    target = 200 if args.pilot else args.limit
    rows = []
    rows.extend(acquire_mutopia(limit=target))
    if len(rows) < target:
        rows.extend(acquire_kunstderfuge(limit=target - len(rows)))

    write_works_csv(rows)
    print(f"[01_acquire] wrote {len(rows)} rows to {WORKS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
