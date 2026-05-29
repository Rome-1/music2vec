"""06_per_projection.py — single-projection large figures, one per method.

Mirrors flag2vec's 06. Useful for Read of the README (each projection
gets its own page-width view) and for spotting cluster structure that
the multi-panel hero compresses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJ_DIR = ROOT / "data" / "projections"
OUT_DIR = ROOT / "out" / "projections"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="audio_mert95")
    ap.add_argument("--no-thumbs", action="store_true",
                    help="force circle markers even if thumbnails exist")
    ap.add_argument("--thumb-px", type=int, default=44,
                    help="thumbnail height in single-projection figures")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    from music2vec.style import (BG, TEXT, SUBTLE, configure_typography,
                                  clean_axes, set_axis_limits, load_thumb)

    proj = pd.read_parquet(PROJ_DIR / f"{args.name}.parquet")
    meta = json.loads((PROJ_DIR / f"{args.name}.meta.json").read_text())
    ev = meta.get("pca_explained_variance_ratio", [0.0, 0.0])
    configure_typography()

    THUMB_DIR = ROOT / "data" / "thumbs"
    n_thumbs = sum(
        1 for w in proj["work_id"]
        if (THUMB_DIR / f"{w}.png").exists()
    )
    use_thumbs = (not args.no_thumbs) and n_thumbs >= len(proj) // 3
    thumb_px = args.thumb_px
    if use_thumbs:
        print(f"[06_per_projection] using score-thumbnail marks "
              f"({n_thumbs}/{len(proj)} available, height={thumb_px}px)")

    panels = [
        ("PCA",   "pca_x",   "pca_y",
         f"{ev[0]*100:.0f}% + {ev[1]*100:.0f}% var"),
        ("t-SNE", "tsne_x",  "tsne_y",  "perplexity 30, cosine"),
        ("UMAP",  "umap_x",  "umap_y",  "n_neighbors 15, cosine"),
        ("PHATE", "phate_x", "phate_y", "knn 15, decay 20"),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, xc, yc, sub in panels:
        if proj[xc].isna().all():
            print(f"  skip {label} (unavailable)")
            continue
        fig, ax = plt.subplots(figsize=(11, 11), facecolor=BG)
        sub_df = proj.dropna(subset=[xc, yc])
        xs = sub_df[xc].values
        ys = sub_df[yc].values
        if use_thumbs:
            # PCA concentrates 38% of variance in two dims and packs the
            # Bach harpsichord cluster into a tiny region — same-sized
            # thumbs that work in t-SNE/UMAP/PHATE turn that corner into
            # a black smudge. Halve the thumb size for PCA.
            panel_px = thumb_px if label != "PCA" else max(20, thumb_px // 2)
            ax.scatter(xs, ys, s=8, color=SUBTLE, alpha=0.35,
                       edgecolor="none", zorder=1)
            for i, work_id in enumerate(sub_df["work_id"].values):
                thumb = load_thumb(work_id, panel_px, max_aspect=1.4)
                if thumb.shape[0] <= 1:
                    continue
                oi = OffsetImage(thumb, zoom=1.0, interpolation="lanczos")
                ax.add_artist(AnnotationBbox(
                    oi, (xs[i], ys[i]),
                    frameon=False, pad=0.0, zorder=3,
                ))
        else:
            ax.scatter(xs, ys, s=18, color=SUBTLE, alpha=0.7,
                       edgecolor="white", linewidth=0.4)
        ax.set_title(f"{label} — {sub}", color=TEXT, fontsize=14, loc="left")
        clean_axes(ax)
        set_axis_limits(ax, xs, ys)
        out_path = OUT_DIR / f"{label.lower().replace('-', '')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
