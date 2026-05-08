"""04_project.py — PCA / t-SNE / UMAP / PHATE projections.

PCA is the hero per Rome's spec (2026-05-08); t-SNE / UMAP / PHATE
secondary panels below.

Reads:  data/embeddings/<name>.npy + .work_ids.txt
Writes: data/projections/<name>.parquet with columns
    work_id, pca_x, pca_y, tsne_x, tsne_y, umap_x, umap_y, phate_x, phate_y
plus data/projections/<name>.meta.json with explained_variance_ratio etc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
PROJ_DIR = ROOT / "data" / "projections"


def project(name: str) -> int:
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        import umap
    except ImportError:
        umap = None
    try:
        import phate
    except ImportError:
        phate = None

    arr_path = EMB_DIR / f"{name}.npy"
    ids_path = EMB_DIR / f"{name}.work_ids.txt"
    if not arr_path.exists() or not ids_path.exists():
        print(f"[04_project] missing {arr_path}; run embed step",
              file=sys.stderr)
        return 1

    X = np.load(arr_path)
    ids = ids_path.read_text().splitlines()
    df = pd.DataFrame({"work_id": ids})

    pca = PCA(n_components=2, random_state=0).fit(X)
    P = pca.transform(X)
    df["pca_x"], df["pca_y"] = P[:, 0], P[:, 1]

    T = TSNE(n_components=2, perplexity=min(30, len(X) // 4),
             metric="cosine", init="pca", random_state=0).fit_transform(X)
    df["tsne_x"], df["tsne_y"] = T[:, 0], T[:, 1]

    if umap is not None:
        U = umap.UMAP(n_components=2, n_neighbors=15, metric="cosine",
                      random_state=0).fit_transform(X)
        df["umap_x"], df["umap_y"] = U[:, 0], U[:, 1]
    else:
        df["umap_x"] = df["umap_y"] = None

    if phate is not None and len(X) >= 30:
        PH = phate.PHATE(n_components=2, knn=15, decay=20,
                         random_state=0, verbose=0).fit_transform(X)
        df["phate_x"], df["phate_y"] = PH[:, 0], PH[:, 1]
    else:
        df["phate_x"] = df["phate_y"] = None

    PROJ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROJ_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    meta = {
        "name": name,
        "n_works": int(len(X)),
        "embedding_dim": int(X.shape[1]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    (PROJ_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[04_project] wrote {out_path} ({len(df)} works)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="audio_mert95",
                    help="Embedding name (matches EMB_DIR file stem)")
    args = ap.parse_args()
    return project(args.name)


if __name__ == "__main__":
    raise SystemExit(main())
