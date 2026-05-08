"""03c_embed_symbolic.py — MusicBERT-on-MIDI symbolic embeddings.

Companion to 03_embed_audio.py. Side-by-side audio-vs-symbolic figures
reveal which structure each modality preserves; the symbolic view should
sharpen contrapuntal-device clustering (fugues, canons) where audio
embeddings smear them with timbre/dynamics.

Output: data/embeddings/symbolic_musicbert.{npy,work_ids.txt}
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
EMB_DIR = ROOT / "data" / "embeddings"
WORKS_CSV = ROOT / "data" / "works.csv"

MODEL_ID = "ruru2701/musicbert-v1.1"
EMB_NAME = "symbolic_musicbert"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not WORKS_CSV.exists():
        print(f"[03c] missing {WORKS_CSV}", file=sys.stderr)
        return 1

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    print("[03c] MusicBERT pipeline: deferred until 03_embed_audio is "
          "validated end-to-end. Symbolic comparison is the second figure, "
          "not the blocker.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
