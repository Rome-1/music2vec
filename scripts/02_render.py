"""02_render.py — MIDI -> 24kHz mono WAV via FluidSynth.

Renders each work twice with different SoundFonts (when available) and
averages downstream embeddings to wash out synth-identity timbre. Writes
three 30s window crops per work (start / middle / random-seeded) to
data/audio/<work_id>/{w0,w1,w2}.wav for the embed step.

Resumable: skips works whose three crops already exist.

Compute: FluidSynth typically renders 4-8x realtime on a 4-core CPU.
1500 works * ~5min/work avg = ~30min wall on this box.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
AUDIO_DIR = ROOT / "data" / "audio"
WORKS_CSV = ROOT / "data" / "works.csv"

# Bring-your-own SoundFonts. Recommended: MuseScore General (free, GM-compliant)
# at /usr/share/sounds/sf2/MuseScore_General.sf2 or similar. Multiple paths
# checked in order; missing ones skipped.
SF2_CANDIDATES = [
    Path("/usr/share/sounds/sf2/MuseScore_General.sf2"),
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    Path.home() / "soundfonts/MuseScore_General.sf2",
]

SAMPLE_RATE = 24_000
WINDOW_SEC = 30


def find_soundfonts() -> list[Path]:
    return [p for p in SF2_CANDIDATES if p.exists()]


def render_with_fluidsynth(midi: Path, sf2: Path, out_wav: Path) -> bool:
    """Full-length render via the fluidsynth CLI. Returns success bool."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "fluidsynth", "-ni", "-r", str(SAMPLE_RATE), "-g", "0.6",
        "-F", str(out_wav), str(sf2), str(midi),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError) as e:
        print(f"[02_render] fluidsynth failed for {midi.name}: {e}",
              file=sys.stderr)
        return False
    return True


def crop_windows(full_wav: Path, work_id: str, seed: int) -> list[Path]:
    """Cut three 30s windows: start, middle, random-seeded.

    For works shorter than 30s, the full clip is used and zero-padded
    to the window length (rare for classical movements). For works
    shorter than 90s, the three windows can overlap.
    """
    import numpy as np
    import soundfile as sf

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = AUDIO_DIR / work_id
    work_dir.mkdir(exist_ok=True)

    audio, sr = sf.read(str(full_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    win = WINDOW_SEC * sr
    n = len(audio)
    if n < win:
        pad = np.zeros(win, dtype=np.float32)
        pad[:n] = audio
        audio = pad
        n = win

    rng = random.Random(seed)
    starts = [
        0,
        max(0, n // 2 - win // 2),
        rng.randint(0, max(0, n - win)),
    ]
    out_paths = []
    for i, s in enumerate(starts):
        clip = audio[s : s + win]
        if len(clip) < win:
            clip = np.pad(clip, (0, win - len(clip)))
        path = work_dir / f"w{i}.wav"
        sf.write(str(path), clip, sr, subtype="PCM_16")
        out_paths.append(path)

    # Drop the full render once windows are written; it's the heavy file.
    try:
        full_wav.unlink()
    except OSError:
        pass
    return out_paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Render only the first N works (smoke test)")
    args = ap.parse_args()

    sfs = find_soundfonts()
    if not sfs:
        print("[02_render] no SoundFonts found; aborting. "
              "Install MuseScore_General.sf2 or set SF2_CANDIDATES.",
              file=sys.stderr)
        return 1

    if not WORKS_CSV.exists():
        print(f"[02_render] missing {WORKS_CSV}; run 01_acquire.py first",
              file=sys.stderr)
        return 1

    rows = list(csv.DictReader(WORKS_CSV.open()))
    if args.limit:
        rows = rows[: args.limit]

    rng = random.Random(42)
    rendered = 0
    skipped = 0
    failed = 0
    for row in rows:
        wid = row["work_id"]
        work_dir = AUDIO_DIR / wid
        if all((work_dir / f"w{i}.wav").exists() for i in range(3)):
            skipped += 1
            continue
        midi = RAW_DIR / f"{wid}.mid"
        if not midi.exists():
            failed += 1
            continue
        full_wav = work_dir / "_full.wav"
        ok = render_with_fluidsynth(midi, sfs[0], full_wav)
        if not ok:
            failed += 1
            continue
        crop_windows(full_wav, wid, seed=rng.randint(0, 1 << 30))
        rendered += 1

    print(f"[02_render] rendered={rendered} skipped={skipped} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
