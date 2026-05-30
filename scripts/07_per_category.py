"""07_per_category.py — per-taxonomy highlight figures with soft hulls.

For each (taxonomy, label), one figure showing how that subset clusters
across PCA / t-SNE / UMAP / PHATE. Multi-panel by design because the
*point* of this figure is to compare cluster shape under different
projections — one of the few "shape-comparison" figures Rome's
feedback called out as legitimately multi-panel.

V3.6 (2026-05-30): score-thumbnail marks removed per Rome's feedback.
Highlighted works are colored dots in the per-label palette; the rest
are small gray dots; soft hull drawn when compactness < 0.65 ×
global mean pairwise distance (the flag2vec honesty constraint).

Outputs: out/categories/<taxonomy>/<label>.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJ_DIR = ROOT / "data" / "projections"
TAX_DIR = ROOT / "data" / "taxonomies"
OUT_DIR = ROOT / "out" / "categories"


def load_labels(taxonomy: str) -> dict[str, str]:
    path = TAX_DIR / f"{taxonomy}.csv"
    if not path.exists():
        return {}
    return {row["work_id"]: row["label"]
            for row in csv.DictReader(path.open())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="audio_mert95",
                    help="Embedding name")
    ap.add_argument("--taxonomy", default=None,
                    help="Limit to a single taxonomy")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from music2vec.style import (
        BG, TEXT, FADE, clean_axes, set_axis_limits,
        configure_typography, soft_hull, category_compactness,
        TAXONOMY_COLORS,
    )
    from music2vec.taxonomies import TAXONOMIES, NICE_NAME

    proj = pd.read_parquet(PROJ_DIR / f"{args.name}.parquet")
    meta = json.loads((PROJ_DIR / f"{args.name}.meta.json").read_text())
    ev = meta.get("pca_explained_variance_ratio", [0.0, 0.0])
    configure_typography()

    panels = [
        ("PCA",   "pca_x",   "pca_y",
         f"{ev[0]*100:.0f}% + {ev[1]*100:.0f}% var"),
        ("t-SNE", "tsne_x",  "tsne_y",  "perplexity 30"),
        ("UMAP",  "umap_x",  "umap_y",  "n_neighbors 15"),
        ("PHATE", "phate_x", "phate_y", "knn 15, decay 20"),
    ]

    taxonomies = ([args.taxonomy] if args.taxonomy
                  else list(TAXONOMIES.keys()))

    for tax in taxonomies:
        labels = load_labels(tax)
        if not labels:
            print(f"  skip {tax} (no labels)")
            continue
        per_label: dict[str, list[str]] = defaultdict(list)
        for wid, lbl in labels.items():
            per_label[lbl].append(wid)

        out_tax_dir = OUT_DIR / tax
        out_tax_dir.mkdir(parents=True, exist_ok=True)
        accent = TAXONOMY_COLORS.get(tax, "#2E5DA5")
        per_label_color = TAXONOMIES[tax]

        for label, wids in per_label.items():
            if len(wids) < 2:
                continue
            color = per_label_color.get(label, accent)
            mask = proj["work_id"].isin(set(wids))

            # 2×2 grid of equal-size panels; cleaner than the old 2-row
            # asymmetric layout once thumbnails are gone.
            fig, axes = plt.subplots(2, 2, figsize=(13, 12),
                                     facecolor=BG)
            fig.subplots_adjust(left=0.05, right=0.97, top=0.92,
                                bottom=0.05, hspace=0.18, wspace=0.10)

            for ax, (plabel, xc, yc, sub) in zip(axes.flat, panels):
                if proj[xc].isna().all():
                    ax.text(0.5, 0.5, f"{plabel} unavailable",
                            transform=ax.transAxes, ha="center",
                            va="center", color=FADE, fontsize=10)
                    clean_axes(ax)
                    continue
                valid = ~proj[xc].isna()
                pxs = proj.loc[valid, xc].values
                pys = proj.loc[valid, yc].values
                m_v = mask & valid
                in_pts = np.stack([proj.loc[m_v, xc].values,
                                   proj.loc[m_v, yc].values], axis=1)
                all_pts = np.stack([pxs, pys], axis=1)
                cp = (category_compactness(in_pts, all_pts)
                      if len(in_pts) >= 2 else 0.0)

                # Background dots — gray, faded, no edge
                bg_mask = valid & (~mask)
                ax.scatter(proj.loc[bg_mask, xc],
                           proj.loc[bg_mask, yc],
                           s=12, color=FADE, alpha=0.55,
                           edgecolor="none", zorder=1)
                # Foreground dots — category color, larger, with edge
                ax.scatter(proj.loc[m_v, xc],
                           proj.loc[m_v, yc],
                           s=58, color=color, alpha=0.92,
                           edgecolor=BG, linewidth=0.9, zorder=3)
                if cp and cp < 0.65 and len(in_pts) >= 3:
                    soft_hull(ax, in_pts, color)

                ax.set_title(
                    f"{plabel}  ·  {sub}  ·  compactness {cp:.2f}",
                    color=TEXT, fontsize=11, loc="left", pad=6,
                )
                clean_axes(ax)
                set_axis_limits(ax, pxs, pys)

            fig.suptitle(
                f"{NICE_NAME.get(tax, tax)} — {label}  ·  n={len(wids)}",
                color=TEXT, fontsize=14, x=0.05, y=0.97, ha="left",
            )

            out_path = out_tax_dir / f"{label}.png"
            fig.savefig(out_path, dpi=180, facecolor=BG)
            plt.close(fig)
            print(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
