"""07_per_category.py — per-taxonomy highlight figures with soft hulls.

For each taxonomy, generate one figure per label that shows the labeled
works highlighted across all four projections (PCA hero up top + t-SNE,
UMAP, PHATE row below), the rest faded to grayscale. Mirrors flag2vec's
07_per_category script. Compactness numbers per panel make it visible
which categories cluster in which projections.

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
        BG, TEXT, SUBTLE, FADE, clean_axes, set_axis_limits,
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

            fig = plt.figure(figsize=(14, 14), facecolor=BG)
            gs = fig.add_gridspec(2, 3, height_ratios=[2, 1],
                                  hspace=0.14, wspace=0.06)

            # Hero PCA on top
            for col_span, (key, idx) in enumerate(zip(panels, range(4))):
                pass
            # Top: PCA spans 3 cols
            ax = fig.add_subplot(gs[0, :])
            xc, yc = panels[0][1], panels[0][2]
            xs, ys = proj[xc].values, proj[yc].values
            in_pts = np.stack([proj.loc[mask, xc].values,
                               proj.loc[mask, yc].values], axis=1)
            all_pts = np.stack([xs, ys], axis=1)
            comp = (category_compactness(in_pts, all_pts)
                    if len(in_pts) >= 2 else 0.0)
            ax.scatter(xs[~mask], ys[~mask], s=10, color=FADE,
                       alpha=0.6, edgecolor="none")
            ax.scatter(xs[mask], ys[mask], s=22, color=color,
                       alpha=0.95, edgecolor="white", linewidth=0.5)
            if comp and comp < 0.65:
                soft_hull(ax, in_pts, color)
            ax.set_title(
                f"{NICE_NAME.get(tax, tax)} — {label}  "
                f"·  PCA  ·  compactness {comp:.2f}"
                f"  ·  n={len(wids)}",
                color=TEXT, fontsize=12, loc="left",
            )
            clean_axes(ax)
            set_axis_limits(ax, xs, ys)

            # Bottom row: t-SNE / UMAP / PHATE
            for col, (plabel, pxc, pyc, psub) in enumerate(panels[1:]):
                if proj[pxc].isna().all():
                    continue
                axb = fig.add_subplot(gs[1, col])
                pxs = proj[pxc].values
                pys = proj[pyc].values
                # Drop nans for axis limits
                valid = ~np.isnan(pxs)
                pxs_v, pys_v = pxs[valid], pys[valid]
                ip = np.stack([proj.loc[mask & valid, pxc].values,
                               proj.loc[mask & valid, pyc].values], axis=1)
                ap_ = np.stack([pxs_v, pys_v], axis=1)
                cp = (category_compactness(ip, ap_)
                      if len(ip) >= 2 else 0.0)
                axb.scatter(pxs_v[~mask[valid]], pys_v[~mask[valid]],
                            s=6, color=FADE, alpha=0.6, edgecolor="none")
                axb.scatter(pxs_v[mask[valid]], pys_v[mask[valid]],
                            s=14, color=color, alpha=0.95,
                            edgecolor="white", linewidth=0.4)
                if cp and cp < 0.65:
                    soft_hull(axb, ip, color)
                axb.set_title(
                    f"{plabel} — compactness {cp:.2f}",
                    color=TEXT, fontsize=10, loc="left",
                )
                clean_axes(axb)
                set_axis_limits(axb, pxs_v, pys_v)

            out_path = out_tax_dir / f"{label}.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight",
                        facecolor=BG)
            plt.close(fig)
            print(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
