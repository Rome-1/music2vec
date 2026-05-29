"""Render a one-system score-thumbnail PNG per work, for use as scatter marks.

For each work in data/works.csv:
  1. Resolve its source .ly file under data/raw/mutopia_src/ via source_url.
  2. Copy to a temp dir and append a header-override block that suppresses
     every header markup field (title / composer / opus / tagline / footer).
  3. Run `lilypond -dpreview -dinclude-book-title-preview=#f --png` to render
     just the first system as a PNG.
  4. Crop to the inked bounding box, downscale to a fixed height,
     save to data/thumbs/<work_id>.png.

LilyPond is single-threaded and slow (~5-15 s per work). The script
parallelises across workers via multiprocessing.

Re-run is idempotent; pass --force to overwrite existing thumbnails.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MUTOPIA_ROOT = DATA / "raw" / "mutopia_src"
THUMBS = DATA / "thumbs"

HEADER_OVERRIDE = """
\\header {
  title = ##f
  subtitle = ##f
  subsubtitle = ##f
  composer = ##f
  arranger = ##f
  opus = ##f
  piece = ##f
  poet = ##f
  meter = ##f
  instrument = ##f
  copyright = ##f
  tagline = ##f
  footer = ##f
}
\\paper {
  indent = 0
  top-margin = 2
  bottom-margin = 2
  left-margin = 2
  right-margin = 2
  print-first-page-number = ##f
  print-page-number = ##f
}
"""


def resolve_ly(source_url: str) -> Path | None:
    """https://www.mutopiaproject.org/ftp/BachJS/BWV846/wtk1-prelude1/wtk1-prelude1.ly
       → data/raw/mutopia_src/BachJS/BWV846/wtk1-prelude1/wtk1-prelude1.ly"""
    marker = "/ftp/"
    idx = source_url.find(marker)
    if idx < 0:
        return None
    suffix = source_url[idx + len(marker):]
    p = MUTOPIA_ROOT / suffix
    return p if p.exists() else None


def render_one(args: tuple[str, str, bool, int]) -> tuple[str, bool, str]:
    """Returns (work_id, success, message)."""
    work_id, source_url, force, target_height = args
    out_path = THUMBS / f"{work_id}.png"
    if out_path.exists() and not force:
        return work_id, True, "skip-exists"

    ly_src = resolve_ly(source_url)
    if ly_src is None:
        return work_id, False, f"no .ly for {source_url}"

    try:
        from PIL import Image
    except ImportError:
        return work_id, False, "Pillow not installed"

    with tempfile.TemporaryDirectory(prefix=f"m2v-thumb-{work_id}-") as td:
        td_path = Path(td)
        wrapped = td_path / "in.ly"
        # Read with errors=replace because some .ly have stray non-utf8 bytes.
        src = ly_src.read_text(encoding="utf-8", errors="replace")
        wrapped.write_text(src + HEADER_OVERRIDE, encoding="utf-8")

        # Many .ly files do `\include "shared.ly"` or `\include "../foo.ly"`.
        # The originals resolve those against the source directory; we tell
        # lilypond to look there via -I.
        ly_dir = ly_src.parent
        try:
            proc = subprocess.run(
                [
                    "lilypond",
                    f"-I{ly_dir}",
                    f"-I{ly_dir.parent}",
                    "-dpreview",
                    "-dinclude-book-title-preview=#f",
                    "-dlog-level=NONE",
                    "--png",
                    "-dresolution=200",
                    "in.ly",
                ],
                cwd=td_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return work_id, False, "lilypond timeout >300s"
        except Exception as e:
            return work_id, False, f"lilypond exception: {e}"
        preview = td_path / "in.preview.png"
        if proc.returncode != 0 or not preview.exists():
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            return work_id, False, f"lilypond failed: {err[-1] if err else 'unknown'}"

        # Crop to inked bounding box, then downscale to fixed height.
        img = Image.open(preview).convert("RGBA")
        # Make transparent background so the bbox detection is clean.
        # The PNG comes out with white background; treat white as transparent.
        import numpy as np

        arr = np.array(img)
        rgb = arr[..., :3]
        is_white = (rgb > 240).all(axis=-1)
        # Set alpha to 0 where white
        arr[..., 3] = np.where(is_white, 0, arr[..., 3])
        img = Image.fromarray(arr)
        bbox = img.getbbox()
        if bbox is None:
            return work_id, False, "blank render"
        img = img.crop(bbox)
        scale = target_height / img.height
        new_w = max(1, int(round(img.width * scale)))
        img = img.resize((new_w, target_height), Image.LANCZOS)

        THUMBS.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return work_id, True, f"{img.size[0]}x{img.size[1]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--height", type=int, default=80,
                    help="output thumbnail height in pixels (default 80)")
    ap.add_argument("--limit", type=int, default=None,
                    help="render only the first N works (debug)")
    args = ap.parse_args()

    with (DATA / "works.csv").open() as f:
        works = [(row["work_id"], row.get("source_url", ""))
                 for row in csv.DictReader(f)]

    if args.limit:
        works = works[: args.limit]

    THUMBS.mkdir(parents=True, exist_ok=True)
    jobs = [(w, u, args.force, args.height) for w, u in works]

    print(f"[12_render_thumbs] {len(jobs)} works, {args.workers} workers, "
          f"height={args.height}px → {THUMBS}")
    t0 = time.time()
    done = ok = 0
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(render_one, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            try:
                work_id, success, msg = fut.result()
            except Exception as e:
                work_id = futs[fut]
                success, msg = False, f"worker crash: {e}"
            done += 1
            if success:
                ok += 1
            else:
                failures.append((work_id, msg))
            if done % 10 == 0 or done == len(jobs):
                print(f"  [{done:3d}/{len(jobs)}] ok={ok} fail={len(failures)} "
                      f"elapsed={time.time()-t0:.0f}s")
    print(f"[12_render_thumbs] done: {ok}/{len(jobs)} ok in {time.time()-t0:.0f}s")
    if failures:
        print("[12_render_thumbs] failures:")
        for w, m in failures[:30]:
            print(f"  {w}: {m}")
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
