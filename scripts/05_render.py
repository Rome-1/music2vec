"""05_render.py — hero figure: cross-projection shape comparison.

This is one of the few figures whose explicit job is to convey *how*
the four embedding methods reshape the corpus differently — so it
stays multi-panel (PCA, t-SNE, UMAP, PHATE side-by-side). Marks are
dots colored by composer (the most-legible single metric in the
corpus) with a small legend; works without a composer label fade to
gray. Era (chronology) drives marker alpha so older works sit slightly
recessed.

V3.6 (2026-05-30): previously rendered score-thumbnails as marks. That
made dense regions illegible and added no signal beyond "this is a
piece of music." Reverted to colored dots per Rome's feedback.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJ_DIR = ROOT / "data" / "projections"
WORKS_CSV = ROOT / "data" / "works.csv"
OUT_DIR = ROOT / "out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="audio_mert95")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    from music2vec.style import (
        BG, TEXT, SUBTLE, FADE, clean_axes, set_axis_limits,
        configure_typography,
    )
    from music2vec.taxonomies import COMPOSER_COLORS, COMPOSER_NICE_NAME

    proj_path = PROJ_DIR / f"{args.name}.parquet"
    if not proj_path.exists():
        print(f"[05_render] missing {proj_path}; run 04_project.py",
              file=sys.stderr)
        return 1

    df = pd.read_parquet(proj_path)
    meta = json.loads((PROJ_DIR / f"{args.name}.meta.json").read_text())
    ev = meta.get("pca_explained_variance_ratio", [0.0, 0.0])

    # works.csv leaves the composer field blank for ~30% of rows; infer
    # from the work_id prefix instead (mutopia uses a Composer/Opus/Piece
    # directory layout that survives into the work_id slug).
    PREFIX_TO_COMPOSER = {
        "bachjs": "BachJS", "beethovenlv": "BeethovenLv",
        "chopinff": "ChopinFF", "dvoraka": "DvorakA",
        "handelgf": "HandelGF", "mozartwa": "MozartWA",
        "vivaldia": "VivaldiA",
    }

    def composer_of(wid: str) -> str:
        return PREFIX_TO_COMPOSER.get(wid.split("-")[0], "")

    df = df.copy()
    df["composer"] = df["work_id"].map(composer_of)

    # Stable composer ordering: by frequency in the corpus, descending.
    counts = df["composer"].value_counts()
    composers = [c for c in counts.index if c]

    configure_typography()
    fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
    gs = fig.add_gridspec(1, 4, wspace=0.04, left=0.04, right=0.86,
                          top=0.90, bottom=0.08)

    panels = [
        ("PCA",   "pca_x",   "pca_y",
         f"{ev[0]*100:.0f}% + {ev[1]*100:.0f}% var"),
        ("t-SNE", "tsne_x",  "tsne_y",  "perplexity 30"),
        ("UMAP",  "umap_x",  "umap_y",  "n_neighbors 15"),
        ("PHATE", "phate_x", "phate_y", "knn 15, decay 20"),
    ]

    for col, (label, xc, yc, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(f"{label}\n{sub}", color=TEXT, fontsize=11,
                     loc="left", pad=8)
        if df[xc].isna().all():
            ax.text(0.5, 0.5, f"{label} unavailable",
                    transform=ax.transAxes, ha="center", va="center",
                    color=SUBTLE, fontsize=10)
            clean_axes(ax)
            continue

        # Unlabeled (gray) underlay first
        unlabeled = df[df["composer"] == ""]
        if len(unlabeled):
            ax.scatter(unlabeled[xc], unlabeled[yc], s=18,
                       color=FADE, alpha=0.55, edgecolor="none",
                       zorder=1)

        for c in composers:
            sub_df = df[df["composer"] == c]
            color = COMPOSER_COLORS.get(c, SUBTLE)
            ax.scatter(sub_df[xc], sub_df[yc], s=28,
                       color=color, alpha=0.82,
                       edgecolor=BG, linewidth=0.6, zorder=2)

        xs = df[xc].dropna().values
        ys = df[yc].dropna().values
        set_axis_limits(ax, xs, ys)
        clean_axes(ax)

    # Composer legend on the right
    handles = [
        Line2D([0], [0], marker="o",
               color="none", markerfacecolor=COMPOSER_COLORS.get(c, SUBTLE),
               markeredgecolor=BG, markersize=8,
               label=f"{COMPOSER_NICE_NAME.get(c, c)} (n={counts[c]})")
        for c in composers
    ]
    fig.legend(
        handles=handles, loc="center right",
        bbox_to_anchor=(0.99, 0.5), frameon=False,
        fontsize=10, labelcolor=TEXT,
        title="Composer", title_fontsize=11,
        alignment="left",
    )

    ENCODER_NAMES = {
        "audio_mert95":  "MERT-v1-95M",
        "audio_mert330": "MERT-v1-330M",
        "audio_muq":     "MuQ-large",
    }
    fig.suptitle(
        f"music2vec — {len(df)} works in audio embedding space "
        f"({ENCODER_NAMES.get(args.name, args.name)})",
        color=TEXT, fontsize=13, x=0.04, y=0.97, ha="left",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "latent_works.png"
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print(f"[05_render] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
