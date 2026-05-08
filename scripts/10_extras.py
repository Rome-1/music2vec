"""10_extras.py — additional analyses mirroring flag2vec/scripts/09.

Adds, beyond what 09_clustering.py already produces:
  • Prototypical work per category (closest to category centroid)
  • Most-distant work pairs in the dataset
  • Closest cross-category neighbors (visual cousins across labels)
  • k-NN purity vs dimensionality (how much 2D loses)
  • Hierarchical dendrogram (Ward on cosine; leaves colored by category)
  • Pairwise distance histogram (same-cat vs different-cat)
  • Title gallery: prototypical work per taxonomy as a reading aid

Outputs all into out/analysis/<taxonomy>/ next to 09's purity bars.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
TAX_DIR = ROOT / "data" / "taxonomies"
WORKS_CSV = ROOT / "data" / "works.csv"
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
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_distances

    from music2vec.style import (BG, TEXT, SUBTLE, FADE, configure_typography)
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
    works_by_id = {r["work_id"]: r for r in csv.DictReader(WORKS_CSV.open())}

    D = cosine_distances(X)
    np.fill_diagonal(D, np.inf)  # so argmin doesn't return self
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Most-distant work pairs (global) ----
    # Use upper-triangle indices to avoid duplicates
    Dt = D.copy()
    np.fill_diagonal(Dt, -np.inf)
    iu = np.triu_indices_from(Dt, k=1)
    flat = Dt[iu]
    order = np.argsort(flat)[::-1][:10]
    pairs = []
    for k in order:
        i, j = iu[0][k], iu[1][k]
        pairs.append({
            "a": ids[i], "b": ids[j],
            "a_title": works_by_id.get(ids[i], {}).get("title", ""),
            "b_title": works_by_id.get(ids[j], {}).get("title", ""),
            "cosine_distance": float(flat[k]),
        })
    (OUT_DIR / "distant_pairs.json").write_text(json.dumps(pairs, indent=2))

    # ---- k-NN purity vs dimensionality ----
    # Take a single high-coverage taxonomy (instrumentation) for the curve
    np.fill_diagonal(D, np.inf)  # restore for purity scan
    inst_labels = load_labels("instrumentation")
    if inst_labels:
        ord_idx = np.array([id_idx[w] for w in inst_labels])
        ord_lbl = np.array([inst_labels[w] for w in inst_labels])
        from sklearn.decomposition import PCA
        cap = min(X.shape[0], X.shape[1])
        dims = [d for d in [2, 5, 10, 25, 50, 100, 200, 384, 768]
                if d <= cap]
        if cap not in dims:
            dims.append(cap)
        purities = []
        for d in dims:
            Xd = (X if d == X.shape[1]
                  else PCA(n_components=d, random_state=0).fit_transform(X))
            Dd = cosine_distances(Xd)
            np.fill_diagonal(Dd, np.inf)
            Ds = Dd[np.ix_(ord_idx, ord_idx)]
            k = min(5, len(ord_idx) - 1)
            nn = np.argsort(Ds, axis=1)[:, :k]
            p = float((ord_lbl[nn] == ord_lbl[:, None]).mean())
            purities.append((d, p))

        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
        ax.set_facecolor(BG)
        ds, ps = zip(*purities)
        ax.plot(ds, ps, marker="o", color="#2E5DA5")
        ax.set_xscale("log")
        ax.set_xlabel("Embedding dimensionality (PCA reduction)", color=TEXT)
        ax.set_ylabel("k=5 NN purity (instrumentation)", color=TEXT)
        ax.set_title(
            "How much structure does 2D lose?  (instrumentation, n=105)",
            color=TEXT, loc="left",
        )
        ax.axhline(1 / 3, ls="--", color=SUBTLE, alpha=0.6,
                   label="chance (1/3)")
        ax.legend(frameon=False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=SUBTLE)
        fig.savefig(OUT_DIR / "knn_purity_vs_dim.png", dpi=160,
                    bbox_inches="tight", facecolor=BG)
        plt.close(fig)

    # ---- Per-taxonomy: prototypical work + cross-category nearest ----
    np.fill_diagonal(D, 0.0)  # restore
    for tax in TAXONOMIES:
        labels = load_labels(tax)
        if not labels:
            continue
        idx_per: dict[str, list[int]] = defaultdict(list)
        for wid, lbl in labels.items():
            if wid in id_idx:
                idx_per[lbl].append(id_idx[wid])
        if not idx_per:
            continue

        protos = {}
        for lbl, idxs in idx_per.items():
            if len(idxs) < 2:
                continue
            sub = X[idxs]
            centroid = sub.mean(axis=0, keepdims=True)
            # cosine-closest member to centroid
            d = cosine_distances(sub, centroid).ravel()
            best = idxs[int(np.argmin(d))]
            protos[lbl] = {
                "work_id": ids[best],
                "title": works_by_id.get(ids[best], {}).get("title", ""),
                "n_members": len(idxs),
            }

        # closest pairs across labels
        all_idxs = [(i, l) for l, lst in idx_per.items() for i in lst]
        cross = []
        for a in range(len(all_idxs)):
            for b in range(a + 1, len(all_idxs)):
                ia, la = all_idxs[a]
                ib, lb = all_idxs[b]
                if la == lb:
                    continue
                cross.append((D[ia, ib], ia, ib, la, lb))
        cross.sort(key=lambda r: r[0])
        cross_neighbors = [{
            "a": ids[ia], "a_label": la,
            "a_title": works_by_id.get(ids[ia], {}).get("title", ""),
            "b": ids[ib], "b_label": lb,
            "b_title": works_by_id.get(ids[ib], {}).get("title", ""),
            "cosine_distance": float(d),
        } for (d, ia, ib, la, lb) in cross[:8]]

        out_t = OUT_DIR / tax
        out_t.mkdir(parents=True, exist_ok=True)
        (out_t / "prototypical.json").write_text(
            json.dumps(protos, indent=2))
        (out_t / "cross_neighbors.json").write_text(
            json.dumps(cross_neighbors, indent=2))

    # ---- Hierarchical dendrogram (Ward on full embedding) ----
    # Color leaves by instrumentation, the highest-coverage taxonomy
    inst = load_labels("instrumentation")
    if inst:
        # Reorder X so labeled rows come first; unlabeled (none in V1)
        # are appended.
        order_ids = [w for w in ids if w in inst]
        idxs = [id_idx[w] for w in order_ids]
        Xs = X[idxs]
        labels_ord = [inst[w] for w in order_ids]
        Ds = cosine_distances(Xs)
        Ds_v = squareform(Ds, checks=False)
        Z = linkage(Ds_v, method="average")  # Ward needs Euclidean
        color_for = {
            "solo_keyboard_harpsichord": "#2E5DA5",
            "solo_keyboard_piano":       "#3F73B8",
            "solo_keyboard_organ":       "#1F7A6B",
            "unaccompanied_string":      "#A04C2C",
        }
        leaf_colors = [color_for.get(l, "#888") for l in labels_ord]

        fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
        ax.set_facecolor(BG)
        # scipy's dendrogram doesn't support per-leaf colors directly,
        # so draw it then color the x-tick labels manually.
        d = dendrogram(Z, labels=order_ids, ax=ax, no_labels=True,
                        link_color_func=lambda k: SUBTLE)
        # underline each leaf with its category color
        leaf_order = d["leaves"]
        for i, leaf in enumerate(leaf_order):
            ax.add_patch(plt.Rectangle((i * 10 - 5, -0.0015),
                                         10, 0.001,
                                         color=leaf_colors[leaf],
                                         alpha=0.95, clip_on=False))
        ax.set_title(
            "Hierarchical clustering (cosine, average linkage)  ·  "
            "leaves colored by instrumentation",
            color=TEXT, loc="left",
        )
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=SUBTLE, labelbottom=False)
        ax.set_xlabel("105 works", color=SUBTLE)
        ax.set_ylabel("cosine distance", color=SUBTLE)
        fig.savefig(OUT_DIR / "dendrogram.png", dpi=160,
                    bbox_inches="tight", facecolor=BG)
        plt.close(fig)

    # ---- Distance histogram: same-cycle vs different-cycle ----
    cyc = load_labels("opus_cycle")
    if cyc:
        cyc_arr = np.array([cyc.get(w, "") for w in ids])
        same, diff = [], []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = D[i, j]
                if cyc_arr[i] and cyc_arr[i] == cyc_arr[j]:
                    same.append(d)
                elif cyc_arr[i] and cyc_arr[j] and cyc_arr[i] != cyc_arr[j]:
                    diff.append(d)
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
        ax.set_facecolor(BG)
        ax.hist(diff, bins=40, color=FADE, alpha=0.85,
                label=f"different cycle (n={len(diff)})", density=True)
        ax.hist(same, bins=40, color="#2E5DA5", alpha=0.6,
                label=f"same cycle (n={len(same)})", density=True)
        ax.set_xlabel("cosine distance", color=TEXT)
        ax.set_ylabel("density", color=TEXT)
        ax.set_title(
            "Pairwise cosine distance — within vs across opus cycles",
            color=TEXT, loc="left",
        )
        ax.legend(frameon=False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=SUBTLE)
        fig.savefig(OUT_DIR / "distance_hist_opus_cycle.png", dpi=160,
                    bbox_inches="tight", facecolor=BG)
        plt.close(fig)

    print("[10_extras] wrote prototypical + cross-neighbor JSON, "
          "knn_purity_vs_dim, dendrogram, distance histogram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
