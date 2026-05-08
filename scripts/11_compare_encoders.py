"""11_compare_encoders.py — side-by-side encoder comparison.

Reads the per-encoder summary.json files produced by 09_clustering and
emits one consolidated table figure showing k-NN purity / k-means
ARI/NMI per taxonomy across all encoders, plus a 3-panel PCA
side-by-side comparing scatter geometry across encoders.

Inputs:
    data/embeddings/<name>.npy
    data/projections/<name>.parquet
    out/analysis_<name>/summary.json   (per-encoder summary written by 09)

NB: 09_clustering currently writes to out/analysis/ regardless of
encoder. To support per-encoder summaries we save them to
out/analysis_<name>/ inside this script before assembling the
comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
PROJ_DIR = ROOT / "data" / "projections"
OUT_DIR = ROOT / "out"


def per_encoder_analysis(name: str) -> Path:
    """Run 09_clustering for `name`, then move the output dir to a
    name-specific path so multiple encoders don't clobber each other."""
    out_path = OUT_DIR / f"analysis_{name}"
    if (out_path / "summary.json").exists():
        return out_path
    print(f"  running 09_clustering for {name} ...")
    subprocess.run(
        ["python3", "scripts/09_clustering.py", name],
        check=True, cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    src = OUT_DIR / "analysis"
    if src.exists():
        if out_path.exists():
            shutil.rmtree(out_path)
        shutil.copytree(src, out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("encoders", nargs="*",
                    default=["audio_mert95", "audio_mert330", "audio_muq"])
    args = ap.parse_args()

    summaries = {}
    for name in args.encoders:
        if not (EMB_DIR / f"{name}.npy").exists():
            print(f"  skip {name} (missing embeddings)")
            continue
        out_path = per_encoder_analysis(name)
        sj = out_path / "summary.json"
        if sj.exists():
            summaries[name] = json.loads(sj.read_text())

    if not summaries:
        print("no encoder summaries produced", file=sys.stderr)
        return 1

    # ---- Comparison table (CSV + markdown) ----
    rows = []
    taxonomies = sorted({tax for s in summaries.values() for tax in s})
    for tax in taxonomies:
        for name, s in summaries.items():
            if tax not in s:
                continue
            r = s[tax]
            rows.append({
                "taxonomy": tax,
                "encoder": name,
                "n_labeled": r["n_labeled"],
                "n_classes": r["n_classes"],
                "knn_purity_k5": round(r["knn_purity_k5"], 3),
                "kmeans_ari": round(r["kmeans_ari"], 3),
                "kmeans_nmi": round(r["kmeans_nmi"], 3),
            })
    out_csv = OUT_DIR / "encoder_comparison.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown table for the README
    md_lines = ["| Taxonomy | Encoder | n | K | k=5 NN purity | ARI | NMI |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        md_lines.append(
            f"| {r['taxonomy']} | {r['encoder']} | {r['n_labeled']} "
            f"| {r['n_classes']} | {r['knn_purity_k5']:.2f} "
            f"| {r['kmeans_ari']:.2f} | {r['kmeans_nmi']:.2f} |"
        )
    (OUT_DIR / "encoder_comparison.md").write_text("\n".join(md_lines))
    print(f"  wrote {out_csv} and {OUT_DIR/'encoder_comparison.md'}")

    # ---- Side-by-side PCA panel ----
    import matplotlib.pyplot as plt
    import pandas as pd

    from music2vec.style import (BG, TEXT, SUBTLE, configure_typography,
                                  clean_axes, set_axis_limits)
    configure_typography()

    fig, axs = plt.subplots(1, len(args.encoders),
                             figsize=(6 * len(args.encoders), 6),
                             facecolor=BG)
    if len(args.encoders) == 1:
        axs = [axs]
    for ax, name in zip(axs, args.encoders):
        pp = PROJ_DIR / f"{name}.parquet"
        if not pp.exists():
            ax.set_visible(False)
            continue
        df = pd.read_parquet(pp)
        meta = json.loads((PROJ_DIR / f"{name}.meta.json").read_text())
        ev = meta.get("pca_explained_variance_ratio", [0.0, 0.0])
        ax.scatter(df["pca_x"], df["pca_y"], s=14, color=SUBTLE,
                   alpha=0.7, edgecolor="none")
        ax.set_title(
            f"{name}  ·  d={meta['embedding_dim']}\n"
            f"PCA  {ev[0]*100:.0f}% + {ev[1]*100:.0f}% var",
            color=TEXT, fontsize=11, loc="left",
        )
        clean_axes(ax)
        set_axis_limits(ax, df["pca_x"].values, df["pca_y"].values)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "encoder_compare_pca.png", dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote {OUT_DIR/'encoder_compare_pca.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
