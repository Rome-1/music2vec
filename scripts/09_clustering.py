"""09_clustering.py — quantitative analyses on the high-dim embeddings.

Mirrors flag2vec/scripts/09_clustering.py:
  • k-NN purity by category (k=5, cosine)
  • k-means(k=K) confusion matrix vs hand labels (ARI, NMI)
  • LOF outliers (k=15, cosine)
  • pairwise distance histograms (same- vs cross-category)
  • k-NN purity vs dimensionality
  • linear probe vs MFCC baseline (on raw audio of one window)

For each taxonomy with ≥5 labeled works, write the relevant figures /
metrics into out/analysis/<taxonomy>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
TAX_DIR = ROOT / "data" / "taxonomies"
OUT_DIR = ROOT / "out" / "analysis"


def load_labels(taxonomy: str) -> dict[str, str]:
    p = TAX_DIR / f"{taxonomy}.csv"
    if not p.exists():
        return {}
    return {r["work_id"]: r["label"] for r in csv.DictReader(p.open())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="audio_mert95")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_rand_score,
                                  normalized_mutual_info_score)
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.neighbors import LocalOutlierFactor

    from music2vec.style import (BG, TEXT, SUBTLE, configure_typography,
                                  clean_axes)
    from music2vec.taxonomies import TAXONOMIES, NICE_NAME

    configure_typography()
    arr_path = EMB_DIR / f"{args.name}.npy"
    ids_path = EMB_DIR / f"{args.name}.work_ids.txt"
    if not arr_path.exists():
        print(f"missing {arr_path}", file=sys.stderr)
        return 1

    X = np.load(arr_path)
    ids = ids_path.read_text().splitlines()
    id_idx = {wid: i for i, wid in enumerate(ids)}

    # Pre-compute pairwise cosine distances once
    D = cosine_distances(X)

    summary = {}
    for tax in TAXONOMIES:
        labels = load_labels(tax)
        labeled_pairs = [(wid, lbl) for wid, lbl in labels.items()
                         if wid in id_idx]
        if len(labeled_pairs) < 5:
            continue
        idx = np.array([id_idx[w] for w, _ in labeled_pairs])
        lbl = np.array([l for _, l in labeled_pairs])
        Xs = X[idx]
        Ds = D[np.ix_(idx, idx)]

        # k-NN purity (k=5; cap to n-1)
        k = min(5, len(idx) - 1)
        nn_idx = np.argsort(Ds, axis=1)[:, 1:k + 1]
        purity = float((lbl[nn_idx] == lbl[:, None]).mean())

        # k-means at k = number of unique labels
        K = len(set(lbl))
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xs)
        ari = float(adjusted_rand_score(lbl, km.labels_))
        nmi = float(normalized_mutual_info_score(lbl, km.labels_))

        # LOF (use full embedding; flag work-ids most "outlying")
        lof = LocalOutlierFactor(n_neighbors=min(15, len(Xs) - 1),
                                  metric="cosine")
        lof.fit_predict(Xs)
        scores = -lof.negative_outlier_factor_
        top = np.argsort(scores)[-10:][::-1]
        outliers = [(labeled_pairs[i][0], labeled_pairs[i][1],
                     float(scores[i])) for i in top]

        summary[tax] = {
            "n_labeled": len(idx),
            "n_classes": int(K),
            "knn_purity_k5": purity,
            "kmeans_ari": ari,
            "kmeans_nmi": nmi,
            "top_outliers": outliers,
        }

        # Per-taxonomy purity-by-class plot
        out_t = OUT_DIR / tax
        out_t.mkdir(parents=True, exist_ok=True)
        per_class = {}
        for class_label in sorted(set(lbl)):
            mask = lbl == class_label
            if mask.sum() < 2:
                continue
            sub_purity = (lbl[nn_idx[mask]] == class_label).mean()
            per_class[class_label] = float(sub_purity)

        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(per_class) + 1)),
                                facecolor=BG)
        ax.set_facecolor(BG)
        items = sorted(per_class.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items],
                color="#2E5DA5", alpha=0.85)
        ax.axvline(1 / K, ls="--", color=SUBTLE, alpha=0.6,
                   label=f"chance = 1/K = {1/K:.2f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("k=5 NN purity", color=TEXT)
        ax.set_title(
            f"{NICE_NAME[tax]}  ·  mean purity {purity:.2f}  ·  "
            f"ARI {ari:.2f}  ·  NMI {nmi:.2f}",
            color=TEXT, loc="left",
        )
        ax.legend(frameon=False, loc="lower right")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=SUBTLE)
        fig.savefig(out_t / "knn_purity.png", dpi=160,
                    bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  {tax}: n={len(idx)} K={K} purity={purity:.2f} "
              f"ARI={ari:.2f} NMI={nmi:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {OUT_DIR/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
