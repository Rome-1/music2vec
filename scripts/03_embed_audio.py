"""03_embed_audio.py — MERT-v1-95M frozen embeddings of rendered audio.

Per Rome (2026-05-08) + compute reality (no GPU, 4 cores, ~10 GiB RAM):
the V1 audio leg uses MERT-v1-95M instead of MERT-v1-330M. Same family,
self-supervised, semantically rich; ~3-5x faster on CPU. Promotion to
330M deferred to cloud-GPU run if v1 results justify.

Pipeline per work:
    load 3x30s windows -> mean over windows -> mean over time
    -> single 768-d vector per work.

Output: data/embeddings/audio_mert95.npy (n_works x 768) and
        data/embeddings/audio_mert95.work_ids.txt (parallel ordering).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
EMB_DIR = ROOT / "data" / "embeddings"
WORKS_CSV = ROOT / "data" / "works.csv"

MODEL_ID = "m-a-p/MERT-v1-95M"
TARGET_SR = 24_000
EMB_NAME = "audio_mert95"


def load_model():
    """Lazy import torch / transformers so the module is importable
    in environments without the model installed."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    return model, extractor


def embed_work(model, extractor, work_dir: Path):
    """Mean-pool 3 windows -> single vector. Stub for CI; real impl below."""
    import numpy as np
    import soundfile as sf
    import torch

    vecs = []
    for i in range(3):
        wav_path = work_dir / f"w{i}.wav"
        if not wav_path.exists():
            continue
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        inputs = extractor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # Mean of last hidden state across time -> 1 x H
        h = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        vecs.append(h)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not WORKS_CSV.exists():
        print(f"[03_embed_audio] missing {WORKS_CSV}", file=sys.stderr)
        return 1

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(WORKS_CSV.open()))
    if args.limit:
        rows = rows[: args.limit]

    print(f"[03_embed_audio] loading {MODEL_ID} ...", file=sys.stderr)
    model, extractor = load_model()

    import numpy as np
    out_vecs, out_ids = [], []
    for row in rows:
        wid = row["work_id"]
        work_dir = AUDIO_DIR / wid
        if not work_dir.exists():
            continue
        vec = embed_work(model, extractor, work_dir)
        if vec is None:
            continue
        out_vecs.append(vec)
        out_ids.append(wid)
        if len(out_ids) % 50 == 0:
            print(f"[03_embed_audio] {len(out_ids)} works embedded",
                  file=sys.stderr)

    if not out_vecs:
        print("[03_embed_audio] no audio found; run 02_render.py first",
              file=sys.stderr)
        return 1

    arr = np.stack(out_vecs).astype("float32")
    np.save(EMB_DIR / f"{EMB_NAME}.npy", arr)
    (EMB_DIR / f"{EMB_NAME}.work_ids.txt").write_text("\n".join(out_ids))
    print(f"[03_embed_audio] wrote {arr.shape} -> {EMB_DIR/EMB_NAME}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
