"""Shared style + helpers for music2vec figures.

Mirrors flag2vec/flag2vec/style.py: warm off-white background, restrained
accents, soft hulls, faded-grayscale highlight figures. Music-domain swaps:
work thumbnails come from rendered score images (first system) rather than
flag PNGs, and the categorical palettes target compositional/dance/sacred/
instrumentation/cycle/national-school taxonomies.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
WORKS_CSV = ROOT / "data" / "works.csv"
TAX_DIR = ROOT / "data" / "taxonomies"
THUMB_DIR = ROOT / "data" / "thumbs"
EMB_DIR = ROOT / "data" / "embeddings"
PROJ_DIR = ROOT / "data" / "projections"
OUT_DIR = ROOT / "out"

# Anthropic-ish warm off-white. Same hex flag2vec uses.
BG = "#F7F4EC"
TEXT = "#1A1A1A"
SUBTLE = "#6B6B6B"
GRID = "#E8E2D2"
FADE = "#C9C4B8"

# Hero hull palette — one color per *taxonomy* (not per category within it),
# so per-taxonomy figures use a single accent color and the taxonomy's
# internal categories are distinguished by hue variation handled in
# taxonomies.py. Keep these distinct, restrained, print-safe.
TAXONOMY_COLORS = {
    "compositional_device": "#2E5DA5",   # cobalt — counterpoint as structure
    "dance_type":           "#A04C2C",   # terracotta — embodied rhythm
    "instrumentation":      "#3E8C73",   # forest — material/texture
    "sacred_function":      "#6E5C9C",   # plum — liturgical
    "opus_cycle":           "#9C8A5C",   # ochre — work-as-set
    "national_school":      "#8C5A2E",   # umber — school/tradition
}

# Projection panel order. PCA hero per Rome's spec; t-SNE / UMAP / PHATE
# below as secondary panels.
PANELS = [
    ("PCA",   "pca_x",   "pca_y",   "explained var injected at render"),
    ("t-SNE", "tsne_x",  "tsne_y",  "perplexity 30, cosine"),
    ("UMAP",  "umap_x",  "umap_y",  "n_neighbors 15, cosine"),
    ("PHATE", "phate_x", "phate_y", "knn 15, decay 20"),
]


def configure_typography() -> None:
    plt.rcParams["font.family"] = [
        "Inter", "IBM Plex Sans", "Helvetica Neue", "Helvetica",
        "Arial", "DejaVu Sans",
    ]
    plt.rcParams["axes.titleweight"] = "medium"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def load_thumb(work_id: str, height_px: int, border_color: str | None = None,
               border_px: int = 3, faded: bool = False,
               grayscale: bool = False, max_aspect: float = 2.2) -> np.ndarray:
    """Load a per-work thumbnail (rendered score first-system PNG).

    Mirrors flag2vec's load_thumb. Falls back to a 1×1 transparent pixel
    if the thumbnail is missing — embed pipelines may run before
    thumbnails are rendered.

    The rendered first system is typically very wide (one full line of
    music). For scatter-plot marks we crop the *left edge* to a maximum
    aspect ratio of `max_aspect`:1 — keeping just the opening measure or
    two so the marker is recognizably-musical without dominating the plot.
    Pass max_aspect=None to preserve the full first system.
    """
    path = THUMB_DIR / f"{work_id}.png"
    if not path.exists():
        return np.zeros((1, 1, 4), dtype=np.uint8)
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    # Truncate to leftmost portion before downscaling, so we keep the
    # opening of the piece (the figure mark) rather than averaging the
    # whole line down to a smudge.
    if max_aspect is not None and img.width > img.height * max_aspect:
        crop_w = max(1, int(round(img.height * max_aspect)))
        img = img.crop((0, 0, crop_w, img.height))
    scale = height_px / img.height
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    if grayscale:
        gray = ImageOps.grayscale(img)
        img = Image.merge("RGBA", (gray, gray, gray, img.split()[-1]))
    if faded:
        alpha = img.split()[-1]
        alpha = alpha.point(lambda v: int(v * 0.28))
        img.putalpha(alpha)
    if border_color:
        img = ImageOps.expand(img, border=border_px, fill=border_color)
    return np.asarray(img)


def soft_hull(ax, pts: np.ndarray, color: str,
              alpha_fill: float = 0.08, alpha_edge: float = 0.30,
              expand: float = 0.04) -> None:
    """Convex-hull soft fill + edge. Skip when fewer than 3 points."""
    if len(pts) < 3:
        return
    try:
        hull = ConvexHull(pts)
    except Exception:
        return
    poly = pts[hull.vertices]
    centroid = poly.mean(axis=0)
    poly = centroid + (poly - centroid) * (1.0 + expand)
    poly = np.vstack([poly, poly[:1]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha_fill,
            zorder=0, linewidth=0)
    ax.plot(poly[:, 0], poly[:, 1], color=color, alpha=alpha_edge,
            linewidth=0.7, zorder=0)


def category_compactness(pts: np.ndarray, all_pts: np.ndarray) -> float:
    """Mean intra-category pairwise distance / mean global pairwise distance.

    Below ~0.65 = visually compact in this projection (flag2vec's threshold).
    """
    if len(pts) < 2:
        return 0.0
    inner = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    inner_mean = inner[np.triu_indices(len(pts), k=1)].mean()
    outer = np.linalg.norm(all_pts[:, None, :] - all_pts[None, :, :], axis=-1)
    outer_mean = outer[np.triu_indices(len(all_pts), k=1)].mean()
    return float(inner_mean / outer_mean) if outer_mean > 0 else 0.0


def clean_axes(ax, bg: str = BG) -> None:
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def set_axis_limits(ax, xs: np.ndarray, ys: np.ndarray,
                    pad_frac: float = 0.07) -> None:
    pad_x = (xs.max() - xs.min()) * pad_frac
    pad_y = (ys.max() - ys.min()) * pad_frac
    ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
