"""05_render.py — hero figure: PCA on top, t-SNE / UMAP / PHATE below.

Rome's layout (2026-05-08): PCA gets the hero panel; t-SNE / UMAP / PHATE
sit below as a row of three secondary panels. Same off-white #F7F4EC
background as flag2vec, soft taxonomy hulls.

When score thumbnails exist under data/thumbs/<work_id>.png the marks
become first-system snippets of the score itself, the way flag2vec
uses the flag as the marker. Falls back to circles otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJ_DIR = ROOT / "data" / "projections"
THUMB_DIR = ROOT / "data" / "thumbs"
OUT_DIR = ROOT / "out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="audio_mert95")
    ap.add_argument("--no-thumbs", action="store_true",
                    help="force circle markers even if thumbnails exist")
    ap.add_argument("--hero-thumb-px", type=int, default=18,
                    help="thumbnail height in the PCA hero panel")
    ap.add_argument("--small-thumb-px", type=int, default=12,
                    help="thumbnail height in the t-SNE/UMAP/PHATE secondary panels")
    ap.add_argument("--thumb-aspect", type=float, default=1.2,
                    help="max width-to-height aspect for thumbnails (lower = more square)")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    from music2vec.style import (
        BG, TEXT, SUBTLE, clean_axes, set_axis_limits,
        configure_typography, load_thumb,
    )

    proj_path = PROJ_DIR / f"{args.name}.parquet"
    if not proj_path.exists():
        print(f"[05_render] missing {proj_path}; run 04_project.py",
              file=sys.stderr)
        return 1

    df = pd.read_parquet(proj_path)
    meta = json.loads((PROJ_DIR / f"{args.name}.meta.json").read_text())
    ev = meta.get("pca_explained_variance_ratio", [0.0, 0.0])

    # Use thumbnails when at least a third of works have one rendered.
    n_thumbs = sum(
        1 for w in df["work_id"]
        if (THUMB_DIR / f"{w}.png").exists()
    )
    use_thumbs = (not args.no_thumbs) and n_thumbs >= len(df) // 3
    if use_thumbs:
        print(f"[05_render] using score-thumbnail marks ({n_thumbs}/{len(df)} available)")
    else:
        print(f"[05_render] using circle marks ({n_thumbs}/{len(df)} thumbs available, "
              f"under threshold or disabled)")

    configure_typography()
    fig = plt.figure(figsize=(14, 14), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.12, wspace=0.06)

    def render_panel(ax, xcol, ycol, thumb_px):
        if df[xcol].notna().any():
            xs = df[xcol].values
            ys = df[ycol].values
            if use_thumbs:
                # Plot a faint reference dot under each thumb so the cloud
                # remains legible where thumbs are missing/very small.
                ax.scatter(xs, ys, s=6, color=SUBTLE,
                           alpha=0.35, edgecolor="none", zorder=1)
                for i, work_id in enumerate(df["work_id"].values):
                    thumb = load_thumb(work_id, thumb_px,
                                       max_aspect=args.thumb_aspect)
                    if thumb.shape[0] <= 1:
                        continue  # missing — leave the dot
                    oi = OffsetImage(thumb, zoom=1.0, interpolation="lanczos")
                    ab = AnnotationBbox(
                        oi, (xs[i], ys[i]),
                        frameon=False, pad=0.0, zorder=3,
                    )
                    ax.add_artist(ab)
            else:
                ax.scatter(xs, ys, s=14, color=SUBTLE,
                           alpha=0.65, edgecolor="none")
            set_axis_limits(ax, xs[~pd.isna(xs)], ys[~pd.isna(ys)])
        else:
            ax.text(0.5, 0.5, f"{xcol} unavailable",
                    transform=ax.transAxes, ha="center", va="center",
                    color=SUBTLE, fontsize=10)
        clean_axes(ax)

    # Hero PCA (top, spans 3 columns)
    ax_pca = fig.add_subplot(gs[0, :])
    ax_pca.set_title(
        f"PCA — {ev[0]*100:.0f}% + {ev[1]*100:.0f}% var",
        color=TEXT, fontsize=13, loc="left",
    )
    render_panel(ax_pca, "pca_x", "pca_y", args.hero_thumb_px)

    # Secondary row: t-SNE / UMAP / PHATE
    for col, (label, xc, yc, sub) in enumerate([
        ("t-SNE", "tsne_x", "tsne_y", "perplexity 30, cosine"),
        ("UMAP",  "umap_x", "umap_y", "n_neighbors 15"),
        ("PHATE", "phate_x", "phate_y", "knn 15, decay 20"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.set_title(f"{label} — {sub}", color=TEXT, fontsize=10, loc="left")
        render_panel(ax, xc, yc, args.small_thumb_px)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "latent_works.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"[05_render] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
