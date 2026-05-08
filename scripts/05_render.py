"""05_render.py — hero figure: PCA on top, t-SNE / UMAP / PHATE below.

Rome's layout (2026-05-08): PCA gets the hero panel; t-SNE / UMAP / PHATE
sit below as a row of three secondary panels. Same off-white #F7F4EC
background as flag2vec, soft taxonomy hulls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJ_DIR = ROOT / "data" / "projections"
OUT_DIR = ROOT / "out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="audio_mert95")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd

    from music2vec.style import (
        BG, TEXT, SUBTLE, clean_axes, set_axis_limits,
        configure_typography, soft_hull, load_thumb,
    )

    proj_path = PROJ_DIR / f"{args.name}.parquet"
    if not proj_path.exists():
        print(f"[05_render] missing {proj_path}; run 04_project.py",
              file=sys.stderr)
        return 1

    df = pd.read_parquet(proj_path)
    meta = json.loads((PROJ_DIR / f"{args.name}.meta.json").read_text())
    ev = meta.get("pca_explained_variance_ratio", [0.0, 0.0])

    configure_typography()
    fig = plt.figure(figsize=(14, 14), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.12, wspace=0.06)

    # Hero PCA (top, spans 3 columns)
    ax_pca = fig.add_subplot(gs[0, :])
    ax_pca.set_title(
        f"PCA — {ev[0]*100:.0f}% + {ev[1]*100:.0f}% var",
        color=TEXT, fontsize=13, loc="left",
    )
    ax_pca.scatter(df["pca_x"], df["pca_y"], s=14,
                   color=SUBTLE, alpha=0.65, edgecolor="none")
    clean_axes(ax_pca)
    set_axis_limits(ax_pca, df["pca_x"].values, df["pca_y"].values)

    # Secondary row: t-SNE / UMAP / PHATE
    for col, (label, xc, yc, sub) in enumerate([
        ("t-SNE", "tsne_x", "tsne_y", "perplexity 30, cosine"),
        ("UMAP",  "umap_x", "umap_y", "n_neighbors 15"),
        ("PHATE", "phate_x", "phate_y", "knn 15, decay 20"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.set_title(f"{label} — {sub}", color=TEXT, fontsize=10, loc="left")
        if df[xc].notna().any():
            ax.scatter(df[xc], df[yc], s=8, color=SUBTLE,
                       alpha=0.6, edgecolor="none")
            set_axis_limits(ax, df[xc].dropna().values,
                            df[yc].dropna().values)
        else:
            ax.text(0.5, 0.5, f"{label} unavailable",
                    transform=ax.transAxes, ha="center", va="center",
                    color=SUBTLE, fontsize=10)
        clean_axes(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "latent_works.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"[05_render] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
