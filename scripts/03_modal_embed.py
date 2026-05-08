"""03_modal_embed.py — local driver for Modal GPU embedding.

Uploads WAVs to the Modal volume (one-time per corpus) and invokes
modal_embed.embed_corpus to compute embeddings on a GPU. Saves the
resulting (n_works × hidden_dim) array to data/embeddings/<name>.npy
next to the local-CPU baseline.

Wire-up:

    # one-time WAV upload (idempotent; modal volume put diffs)
    python3 scripts/03_modal_embed.py --upload-only

    # embed with MERT-v1-330M
    python3 scripts/03_modal_embed.py m-a-p/MERT-v1-330M audio_mert330

    # embed with MuQ-large
    python3 scripts/03_modal_embed.py OpenGVLab/MuQ-large audio_muq

The pre-existing local CPU MERT-95M baseline at audio_mert95.npy is
preserved so we can compare encoders directly in subsequent figures.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
EMB_DIR = ROOT / "data" / "embeddings"
WORKS_CSV = ROOT / "data" / "works.csv"

# modal_embed.py lives at the project root; ensure it's importable
sys.path.insert(0, str(ROOT))


def upload_audio_to_modal() -> None:
    """Sync data/audio/ to the Modal volume at /audio/."""
    import modal
    from modal_embed import VOLUME_NAME

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    print(f"[03_modal] syncing {AUDIO_DIR} -> volume {VOLUME_NAME}:/audio "
          f"(this takes a minute on first run; resumable)")

    work_dirs = sorted(AUDIO_DIR.iterdir())
    n_files = 0
    with vol.batch_upload(force=True) as batch:
        for wd in work_dirs:
            if not wd.is_dir():
                continue
            for wav in wd.glob("w*.wav"):
                rel = f"audio/{wd.name}/{wav.name}"
                batch.put_file(str(wav), rel)
                n_files += 1
    print(f"[03_modal] uploaded {n_files} WAVs across {len(work_dirs)} works")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", nargs="?", default="m-a-p/MERT-v1-330M")
    ap.add_argument("name", nargs="?", default="audio_mert330")
    ap.add_argument("--upload-only", action="store_true",
                    help="Sync WAVs to Modal volume and exit")
    args = ap.parse_args()

    if args.upload_only:
        upload_audio_to_modal()
        return 0

    if not WORKS_CSV.exists():
        print(f"missing {WORKS_CSV}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(WORKS_CSV.open()))
    work_ids = [r["work_id"] for r in rows]

    print(f"[03_modal] {len(work_ids)} works -> {args.model_id} on Modal A10G",
          file=sys.stderr)

    import numpy as np
    from modal_embed import app, embed_corpus

    with app.run():
        result = embed_corpus.remote(args.model_id, work_ids)

    valid_ids = [w for w in work_ids if w in result]
    if not valid_ids:
        print("[03_modal] no embeddings returned", file=sys.stderr)
        return 1

    arr = np.stack([result[w] for w in valid_ids]).astype("float32")

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_DIR / f"{args.name}.npy", arr)
    (EMB_DIR / f"{args.name}.work_ids.txt").write_text("\n".join(valid_ids))
    print(f"[03_modal] wrote {arr.shape} -> {EMB_DIR/args.name}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
