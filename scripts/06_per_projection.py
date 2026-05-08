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
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd

    from music2vec.style import (BG, TEXT, SUBTLE, configure_typography,
                                  clean_axes, set_axis_limits)

    proj = pd.read_parquet(PROJ_DIR / f"{args.name}.parquet")
    meta = json.loads((PROJ_DIR / f"{args.name}.meta.json").read_text())
    ev = meta.get("pca_explained_variance_ratio", [0.0, 0.0])
    configure_typography()

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
        xs = proj[xc].dropna().values
        ys = proj[yc].dropna().values
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
