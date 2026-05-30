"""06_per_projection.py — one single-panel figure per embedding method.

Each is page-width: dots colored by composer, with a legend and a few
hand-picked annotation callouts to anchor the eye. Marker alpha lower
for less-frequent composers so the dominant Bach cluster doesn't drown
out Vivaldi or Mozart.

V3.6 (2026-05-30): score-thumbnail marks removed per Rome's feedback —
they were noise at small sizes and gave the figures a uniform black
texture. Reverted to colored dots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJ_DIR = ROOT / "data" / "projections"
OUT_DIR = ROOT / "out" / "projections"

PREFIX_TO_COMPOSER = {
    "bachjs": "BachJS", "beethovenlv": "BeethovenLv",
    "chopinff": "ChopinFF", "dvoraka": "DvorakA",
    "handelgf": "HandelGF", "mozartwa": "MozartWA",
    "vivaldia": "VivaldiA",
}


def composer_of(wid: str) -> str:
    return PREFIX_TO_COMPOSER.get(wid.split("-")[0], "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="audio_mert95")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    from music2vec.style import (BG, TEXT, SUBTLE, FADE, configure_typography,
                                  clean_axes, set_axis_limits)
    from music2vec.taxonomies import COMPOSER_COLORS, COMPOSER_NICE_NAME

    proj = pd.read_parquet(PROJ_DIR / f"{args.name}.parquet")
    meta = json.loads((PROJ_DIR / f"{args.name}.meta.json").read_text())
    ev = meta.get("pca_explained_variance_ratio", [0.0, 0.0])
    configure_typography()

    proj = proj.copy()
    proj["composer"] = proj["work_id"].map(composer_of)
    counts = proj["composer"].value_counts()
    composers = [c for c in counts.index if c]

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
        # Reserve right margin for legend
        fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
        fig.subplots_adjust(left=0.06, right=0.78, top=0.92, bottom=0.06)

        valid = proj.dropna(subset=[xc, yc])
        # Unlabeled underlay
        unl = valid[valid["composer"] == ""]
        if len(unl):
            ax.scatter(unl[xc], unl[yc], s=22,
                       color=FADE, alpha=0.55, edgecolor="none", zorder=1,
                       label="_nolabel_")

        for c in composers:
            sub_df = valid[valid["composer"] == c]
            color = COMPOSER_COLORS.get(c, SUBTLE)
            ax.scatter(sub_df[xc], sub_df[yc], s=44,
                       color=color, alpha=0.82,
                       edgecolor=BG, linewidth=0.8, zorder=2)

        ax.set_title(f"{label} — {sub}", color=TEXT, fontsize=15,
                     loc="left", pad=10)
        clean_axes(ax)
        set_axis_limits(ax, valid[xc].values, valid[yc].values)

        # Composer legend
        handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COMPOSER_COLORS.get(c, SUBTLE),
                   markeredgecolor=BG, markersize=10,
                   label=f"{COMPOSER_NICE_NAME.get(c, c)} (n={counts[c]})")
            for c in composers
        ]
        fig.legend(
            handles=handles, loc="center right",
            bbox_to_anchor=(0.99, 0.5), frameon=False,
            fontsize=11, labelcolor=TEXT,
            title="Composer", title_fontsize=12, alignment="left",
        )

        out_path = OUT_DIR / f"{label.lower().replace('-', '')}.png"
        fig.savefig(out_path, dpi=200, facecolor=BG)
        plt.close(fig)
        print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
