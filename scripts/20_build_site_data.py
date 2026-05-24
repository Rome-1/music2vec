"""Build the unified dataset that powers the GitHub Pages site.

Produces:
  docs/data/works.json    — every work, with metadata, taxonomy labels,
                            and per-encoder (MERT-95M / MERT-330M / MuQ)
                            3D coords for PCA, t-SNE, UMAP, PHATE.
  docs/audio/<id>.mp3    — 30 s mono preview per work, 96 kbps.

The 2D projections in data/projections/*.parquet already give us PCA/tSNE/UMAP
in 2D; we recompute 3D from the raw .npy embeddings for the depth axis.
Re-running is idempotent — existing mp3 transcodes are skipped.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"

EMB_DIR = DATA / "embeddings"
AUDIO_DIR = DATA / "audio"
DOCS_AUDIO = DOCS / "audio"
DOCS_DATA = DOCS / "data"

DOCS_AUDIO.mkdir(parents=True, exist_ok=True)
DOCS_DATA.mkdir(parents=True, exist_ok=True)

ENCODERS = ["audio_mert95", "audio_mert330", "audio_muq"]
ENCODER_LABELS = {
    "audio_mert95":  "MERT-95M",
    "audio_mert330": "MERT-330M",
    "audio_muq":     "MuQ-large",
}

TAXONOMIES = [
    "compositional_device",
    "dance_type",
    "instrumentation",
    "national_school",
    "opus_cycle",
    "sacred_function",
]

# Composer inference from work_id prefix — works.csv has many NaN composers.
COMPOSER_PREFIXES = {
    "bachjs":     ("Bach",      "Johann Sebastian Bach"),
    "beethovenlv":("Beethoven", "Ludwig van Beethoven"),
    "chopinff":   ("Chopin",    "Frédéric Chopin"),
    "mozartwa":   ("Mozart",    "Wolfgang Amadeus Mozart"),
    "dvoraka":    ("Dvořák",    "Antonín Dvořák"),
    "vivaldia":   ("Vivaldi",   "Antonio Vivaldi"),
}

ERA_BY_COMPOSER = {
    "Vivaldi":  "baroque",
    "Bach":     "baroque",
    "Mozart":   "classical",
    "Beethoven":"classical",
    "Chopin":   "romantic",
    "Dvořák":   "romantic",
}


def composer_of(work_id: str) -> tuple[str, str]:
    """Return (short_name, full_name) inferred from the work_id prefix."""
    head = work_id.split("-", 1)[0].lower()
    return COMPOSER_PREFIXES.get(head, ("Unknown", "Unknown"))


# ─── Load metadata + taxonomies ───────────────────────────────────────────
print("loading metadata…")

works = pd.read_csv(DATA / "works.csv").set_index("work_id")
tax_frames = {}
for tax in TAXONOMIES:
    p = DATA / "taxonomies" / f"{tax}.csv"
    if p.exists():
        tax_frames[tax] = pd.read_csv(p).set_index("work_id")["label"]

# ─── Load embeddings + compute 3D projections per encoder ─────────────────
encoder_coords = {}  # encoder → work_id → coord dict

def normalize_coords(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.mean(0, keepdims=True)
    scale = np.percentile(np.abs(arr), 99)
    return (arr / max(scale, 1e-9)).astype(np.float32)


for enc in ENCODERS:
    emb_path = EMB_DIR / f"{enc}.npy"
    order_path = EMB_DIR / f"{enc}.work_ids.txt"
    if not emb_path.exists() or not order_path.exists():
        print(f"  skipping {enc} (missing files)")
        continue
    print(f"loading {enc}…")
    emb = normalize(np.load(emb_path))
    order = order_path.read_text().splitlines()
    assert len(order) == emb.shape[0]
    print(f"  {emb.shape}, computing 3D PCA / t-SNE / UMAP…")

    pca3 = normalize_coords(PCA(n_components=3, random_state=0).fit_transform(emb))
    tsne3 = normalize_coords(
        TSNE(n_components=3, perplexity=min(30, max(5, emb.shape[0] // 4)),
             metric="cosine", init="pca", learning_rate="auto",
             random_state=0, max_iter=1500).fit_transform(emb)
    )
    import umap
    umap3 = normalize_coords(
        umap.UMAP(n_components=3, metric="cosine", n_neighbors=15,
                  min_dist=0.05, random_state=0).fit_transform(emb)
    )

    coords = {}
    for i, wid in enumerate(order):
        coords[wid] = {
            "pca3":  [float(pca3[i][0]),  float(pca3[i][1]),  float(pca3[i][2])],
            "tsne3": [float(tsne3[i][0]), float(tsne3[i][1]), float(tsne3[i][2])],
            "umap3": [float(umap3[i][0]), float(umap3[i][1]), float(umap3[i][2])],
        }
    # Add 2D from the parquet for the "flat map" toggle.
    parq = DATA / "projections" / f"{enc}.parquet"
    if parq.exists():
        df = pd.read_parquet(parq)
        df = df.set_index("work_id")
        for wid in coords:
            if wid in df.index:
                row = df.loc[wid]
                coords[wid]["pca2"]  = [float(row["pca_x"]),  float(row["pca_y"])]
                coords[wid]["tsne2"] = [float(row["tsne_x"]), float(row["tsne_y"])]
                coords[wid]["umap2"] = [float(row["umap_x"]), float(row["umap_y"])]
    encoder_coords[enc] = coords


# ─── Audio transcode: w1.wav → docs/audio/<id>.mp3 (96 kbps mono) ─────────
print("transcoding audio (skipping existing)…")
n_done, n_skip, n_missing = 0, 0, 0
for wid in works.index:
    src = AUDIO_DIR / wid / "w1.wav"
    dst = DOCS_AUDIO / f"{wid}.mp3"
    if not src.exists():
        n_missing += 1
        continue
    if dst.exists() and dst.stat().st_size > 0:
        n_skip += 1
        continue
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-ac", "1", "-c:a", "libmp3lame",
        "-b:a", "96k", "-ar", "24000",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    n_done += 1
print(f"  transcoded {n_done}, skipped {n_skip}, missing {n_missing}")


# ─── Build the JSON record list ───────────────────────────────────────────
print("building works.json…")

def short_title(t: str | float) -> str:
    if pd.isna(t):
        return ""
    return str(t)

records = []
all_composers = set()
all_eras = set()
all_taxvals = {tax: set() for tax in TAXONOMIES}

for wid in works.index:
    w = works.loc[wid]
    cname_short, cname_full = composer_of(wid)
    composer = w.get("composer")
    if pd.isna(composer):
        composer = cname_short
    else:
        composer = str(composer).replace("BachJS", "Bach").replace("BeethovenLv", "Beethoven") \
                                  .replace("ChopinFF", "Chopin").replace("MozartWA", "Mozart") \
                                  .replace("DvorakA", "Dvořák").replace("VivaldiA", "Vivaldi")

    era = w.get("era")
    if pd.isna(era):
        era = ERA_BY_COMPOSER.get(composer)
    if era and not isinstance(era, str):
        era = None

    rec = {
        "id": wid,
        "composer": composer,
        "composer_full": cname_full,
        "title": short_title(w.get("title")),
        "year": int(w["year"]) if not pd.isna(w.get("year")) else None,
        "instrumentation": short_title(w.get("instrumentation_hint")),
        "era": era,
        "source_url": short_title(w.get("source_url")),
        "audio": f"audio/{wid}.mp3",
        "coords": {enc: encoder_coords[enc].get(wid, {}) for enc in encoder_coords},
    }
    # Taxonomy labels
    rec["tax"] = {}
    for tax, series in tax_frames.items():
        if wid in series.index:
            v = series.loc[wid]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            if pd.notna(v):
                rec["tax"][tax] = str(v)
                all_taxvals[tax].add(str(v))

    all_composers.add(composer)
    if era: all_eras.add(era)
    records.append(rec)


# Filter out works that don't have coords in any encoder.
records = [r for r in records if any(r["coords"].get(e) for e in ENCODERS)]
print(f"  kept {len(records)} works with coords")


# ─── Assemble + write ─────────────────────────────────────────────────────
doc = {
    "meta": {
        "n_works": len(records),
        "encoders": [{"id": e, "label": ENCODER_LABELS[e]} for e in ENCODERS if e in encoder_coords],
        "composers": sorted(all_composers),
        "eras": sorted(all_eras),
        "taxonomies": {tax: sorted(vals) for tax, vals in all_taxvals.items() if vals},
    },
    "works": records,
}

out = DOCS_DATA / "works.json"
out.write_text(json.dumps(doc, separators=(",", ":")))
print(f"  wrote {out} ({out.stat().st_size / 1024:.1f} KB)")

print("\nsummary:")
print(f"  works: {len(records)}")
print(f"  encoders: {list(encoder_coords)}")
print(f"  composers: {sorted(all_composers)}")
print(f"  eras: {sorted(all_eras)}")
for tax, vals in all_taxvals.items():
    if vals:
        print(f"  {tax}: {len(vals)} labels")
print(f"  audio: {len(list(DOCS_AUDIO.glob('*.mp3')))} mp3s, "
      f"{sum(p.stat().st_size for p in DOCS_AUDIO.glob('*.mp3')) / 1024 / 1024:.1f} MB total")
