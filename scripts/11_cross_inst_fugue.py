"""
Cross-instrumentation fugue probe (V4 roadmap item #2).

The V1 README claimed "fugues cluster across instrumentation" based on the
solo-violin fugues from BWV 1001/1003/1005 sitting in the same outer hull
as the WTC keyboard fugues. That was a soft visual observation. This
script turns it into a single number.

Question: do the *non-keyboard* fugues sit nearer to the keyboard fugues
than to non-fugue works in their own instrumentation?

Per-fugue version: for each non-keyboard fugue f, compute
    d_cross = mean cosine distance to keyboard fugues
    d_own   = mean cosine distance to non-fugue works in f's own
              instrumentation
The fugue is "pulled cross-instrument" iff d_cross < d_own.

Pooled version: cross-instrumentation NN purity. For each fugue, restrict
its k=5 nearest neighbors to works of *other* instrumentation classes.
What fraction of those cross-instrument neighbors are fugues?

Outputs:
    out/analysis/<encoder>/cross_inst_fugue.json   per-encoder numbers
    out/analysis/cross_inst_fugue.png              all-encoders bar figure
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"

ENCODERS = ["audio_mert95", "audio_mert330", "audio_muq"]
KEYBOARD = {"solo_keyboard_harpsichord", "solo_keyboard_piano"}


def load(encoder: str):
    emb_path = DATA / "embeddings" / f"{encoder}.npy"
    ids_path = DATA / "embeddings" / f"{encoder}.work_ids.txt"
    emb = np.load(emb_path).astype(np.float32)
    ids = ids_path.read_text().splitlines()
    return emb, ids


def cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = emb / norms
    sim = unit @ unit.T
    return 1.0 - sim


def load_label_map(name: str) -> dict[str, str]:
    out = {}
    with (DATA / "taxonomies" / f"{name}.csv").open() as f:
        for row in csv.DictReader(f):
            out[row["work_id"]] = row["label"]
    return out


def per_fugue_distances(
    emb: np.ndarray,
    ids: list[str],
    fugues: set[str],
    instr: dict[str, str],
) -> list[dict]:
    d_mat = cosine_distance_matrix(emb)
    idx_by_id = {w: i for i, w in enumerate(ids)}

    keyboard_fugues = [w for w in fugues if instr.get(w) in KEYBOARD]
    non_keyboard_fugues = [w for w in fugues if instr.get(w) not in KEYBOARD]
    kf_idx = np.array([idx_by_id[w] for w in keyboard_fugues])

    rows = []
    for w in non_keyboard_fugues:
        i = idx_by_id[w]
        d_cross = float(d_mat[i, kf_idx].mean())
        own_inst = instr.get(w)
        own_inst_non_fugue_idx = np.array(
            [
                idx_by_id[w2]
                for w2 in ids
                if w2 != w
                and instr.get(w2) == own_inst
                and w2 not in fugues
            ]
        )
        d_own = (
            float(d_mat[i, own_inst_non_fugue_idx].mean())
            if len(own_inst_non_fugue_idx)
            else float("nan")
        )
        rows.append(
            {
                "work_id": w,
                "instrumentation": own_inst,
                "d_to_keyboard_fugues": d_cross,
                "d_to_own_nonfugues": d_own,
                "pulled_cross_instrument": d_cross < d_own
                if not np.isnan(d_own)
                else None,
            }
        )
    return rows


def cross_instrument_nn_purity(
    emb: np.ndarray,
    ids: list[str],
    fugues: set[str],
    instr: dict[str, str],
    k: int = 5,
) -> dict:
    """For each fugue: restrict neighborhood to other-instrumentation works,
    measure fugue-share."""
    d_mat = cosine_distance_matrix(emb)
    idx_by_id = {w: i for i, w in enumerate(ids)}

    per_fugue = []
    for w in sorted(fugues):
        i = idx_by_id[w]
        my_inst = instr.get(w)
        # candidate indices: other-instrumentation works
        candidate_ids = [
            w2 for w2 in ids if w2 != w and instr.get(w2) != my_inst
        ]
        if len(candidate_ids) < k:
            continue
        cand_idx = np.array([idx_by_id[w2] for w2 in candidate_ids])
        d = d_mat[i, cand_idx]
        order = np.argsort(d)[:k]
        neighbors = [candidate_ids[j] for j in order]
        fugue_share = sum(1 for n in neighbors if n in fugues) / k
        per_fugue.append(
            {
                "work_id": w,
                "instrumentation": my_inst,
                "cross_inst_fugue_share_k5": fugue_share,
                "cross_inst_neighbors": neighbors,
            }
        )

    if not per_fugue:
        return {"per_fugue": [], "mean": float("nan"), "chance": float("nan")}

    mean_share = float(np.mean([r["cross_inst_fugue_share_k5"] for r in per_fugue]))

    # Chance baseline: pick k random other-instrumentation works. Per-fugue chance
    # is len(other-inst fugues) / len(other-inst works); average across fugues.
    chances = []
    n_works = len(ids)
    for w in sorted(fugues):
        my_inst = instr.get(w)
        others = [w2 for w2 in ids if w2 != w and instr.get(w2) != my_inst]
        if len(others) < k:
            continue
        other_fugues = [w2 for w2 in others if w2 in fugues]
        chances.append(len(other_fugues) / len(others))
    chance = float(np.mean(chances)) if chances else float("nan")

    return {
        "per_fugue": per_fugue,
        "mean_cross_inst_fugue_share_k5": mean_share,
        "chance_cross_inst_fugue_share_k5": chance,
        "lift_over_chance": mean_share / chance if chance else float("nan"),
    }


def run_encoder(encoder: str) -> dict:
    emb, ids = load(encoder)
    instr = load_label_map("instrumentation")
    comp = load_label_map("compositional_device")
    fugues = {w for w, lab in comp.items() if lab == "fugue"}
    fugues &= set(ids)  # only those we have embeddings for

    counts_by_instr = defaultdict(int)
    for w in fugues:
        counts_by_instr[instr.get(w, "?")] += 1

    per_fugue = per_fugue_distances(emb, ids, fugues, instr)
    cross_purity = cross_instrument_nn_purity(emb, ids, fugues, instr, k=5)

    return {
        "encoder": encoder,
        "n_fugues_total": len(fugues),
        "fugue_counts_by_instrumentation": dict(counts_by_instr),
        "per_non_keyboard_fugue": per_fugue,
        "n_pulled_cross_instrument": sum(
            1 for r in per_fugue if r.get("pulled_cross_instrument")
        ),
        "n_non_keyboard_fugues": len(per_fugue),
        "cross_instrumentation_nn_purity_k5": cross_purity,
    }


def render(results: list[dict], out_path: Path) -> None:
    """Two-panel figure: per-fugue distance comparison + cross-inst NN-purity bar."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 2]}
    )

    # Panel 1: per-fugue d_cross vs d_own, grouped by encoder.
    encoders = [r["encoder"] for r in results]
    # union of non-keyboard fugue ids across encoders
    all_works = []
    for r in results:
        for row in r["per_non_keyboard_fugue"]:
            if row["work_id"] not in all_works:
                all_works.append(row["work_id"])

    n_works = len(all_works)
    n_enc = len(encoders)
    x = np.arange(n_works)
    w = 0.18
    colors_cross = ["#2f72b3", "#3aa372", "#c46a3d"]
    colors_own = ["#bcd2e6", "#bedfb4", "#e8c8a8"]
    for ei, r in enumerate(results):
        d = {row["work_id"]: row for row in r["per_non_keyboard_fugue"]}
        ds_cross = [d[w_id]["d_to_keyboard_fugues"] for w_id in all_works]
        ds_own = [d[w_id]["d_to_own_nonfugues"] for w_id in all_works]
        offset = (ei - (n_enc - 1) / 2) * 2 * w
        ax1.bar(
            x + offset - w / 2,
            ds_cross,
            width=w * 0.9,
            color=colors_cross[ei],
            label=f"{r['encoder']}: → keyboard fugues",
        )
        ax1.bar(
            x + offset + w / 2,
            ds_own,
            width=w * 0.9,
            color=colors_own[ei],
            label=f"{r['encoder']}: → own-inst non-fugues",
        )

    def short(w_id: str) -> str:
        # bachjs-bwv1001-bwv-1001-2-bwv-1001-2 → BWV 1001 #2
        parts = w_id.split("-")
        bwv = next((p for p in parts if p.startswith("bwv")), parts[0])
        last_int = next(
            (p for p in reversed(parts) if p.isdigit() or p[:1].isdigit()),
            "",
        )
        return f"{bwv.upper()} #{last_int.lstrip('0') or last_int}"

    ax1.set_xticks(x)
    ax1.set_xticklabels([short(w_id) for w_id in all_works], rotation=20, ha="right")
    ax1.set_ylabel("cosine distance")
    ax1.set_title("Non-keyboard fugues: distance to keyboard fugues vs to own-inst non-fugues")
    ax1.legend(fontsize=8, ncol=2, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: cross-instrumentation NN purity per encoder.
    means = [
        r["cross_instrumentation_nn_purity_k5"]["mean_cross_inst_fugue_share_k5"]
        for r in results
    ]
    chances = [
        r["cross_instrumentation_nn_purity_k5"]["chance_cross_inst_fugue_share_k5"]
        for r in results
    ]
    xpos = np.arange(len(encoders))
    ax2.bar(
        xpos - 0.18, means, width=0.34, color="#3a3a8a", label="observed cross-inst NN purity",
    )
    ax2.bar(
        xpos + 0.18, chances, width=0.34, color="#bababa", label="chance",
    )
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(encoders, rotation=10)
    ax2.set_ylabel("k=5 NN purity (cross-instrument neighbors only)")
    ax2.set_title("Cross-instrument fugue NN purity")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)
    for ei, (m, c) in enumerate(zip(means, chances)):
        if c > 0:
            lift = m / c
            ax2.text(
                ei + 0.18,
                c + 0.01,
                f"{lift:.1f}×",
                ha="center",
                fontsize=9,
                color="#444",
            )

    fig.suptitle(
        "Cross-instrumentation fugue probe — do non-keyboard fugues cluster with keyboard fugues?",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main() -> None:
    results = []
    for encoder in ENCODERS:
        r = run_encoder(encoder)
        out_dir = OUT / f"analysis_{encoder}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "cross_inst_fugue.json").open("w") as f:
            json.dump(r, f, indent=2)
        print(
            f"{encoder}: {r['n_non_keyboard_fugues']} non-keyboard fugues; "
            f"pulled cross-inst: {r['n_pulled_cross_instrument']}; "
            f"cross-inst NN purity k5 = "
            f"{r['cross_instrumentation_nn_purity_k5']['mean_cross_inst_fugue_share_k5']:.2f} "
            f"(chance {r['cross_instrumentation_nn_purity_k5']['chance_cross_inst_fugue_share_k5']:.2f}, "
            f"lift {r['cross_instrumentation_nn_purity_k5']['lift_over_chance']:.1f}×)"
        )
        results.append(r)

    render(results, OUT / "analysis" / "cross_inst_fugue.png")
    with (OUT / "analysis" / "cross_inst_fugue_summary.json").open("w") as f:
        json.dump(
            {
                r["encoder"]: {
                    "n_fugues_total": r["n_fugues_total"],
                    "fugue_counts_by_instrumentation": r[
                        "fugue_counts_by_instrumentation"
                    ],
                    "n_non_keyboard_fugues": r["n_non_keyboard_fugues"],
                    "n_pulled_cross_instrument": r["n_pulled_cross_instrument"],
                    "mean_cross_inst_fugue_share_k5": r[
                        "cross_instrumentation_nn_purity_k5"
                    ]["mean_cross_inst_fugue_share_k5"],
                    "chance_cross_inst_fugue_share_k5": r[
                        "cross_instrumentation_nn_purity_k5"
                    ]["chance_cross_inst_fugue_share_k5"],
                    "lift_over_chance": r["cross_instrumentation_nn_purity_k5"][
                        "lift_over_chance"
                    ],
                }
                for r in results
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
