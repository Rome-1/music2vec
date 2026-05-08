"""modal_embed.py — GPU-hosted MERT/MuQ embedding service.

Per Rome (2026-05-08): Modal greenlit with a hard $5 cap. This module
defines the Modal app + image; the local driver in
scripts/03_modal_embed.py uploads WAVs and pulls embeddings back.

Two encoders supported:
  • m-a-p/MERT-v1-330M    — primary; pitch's original choice
  • OpenGVLab/MuQ-large   — Dec 2024 SOTA on MARBLE; comparison

Wire-up:
  modal volume put music2vec-data data/audio /audio   # one-time WAV upload
  python3 scripts/03_modal_embed.py m-a-p/MERT-v1-330M audio_mert330
  python3 scripts/03_modal_embed.py OpenGVLab/MuQ-large    audio_muq

Cost note: A10G is ~$1.10/hr. 105 works × 3×30s clips = 315 forward
passes ≈ 16 s on GPU per encoder. Even with 30 s container startup +
model download, expected cost is <$0.10 per encoder run. The
batch_size knob lets us trade GPU util for memory headroom; default
keeps clips small to fit in 24 GB.
"""
from __future__ import annotations

import modal

APP_NAME = "music2vec-embed"
VOLUME_NAME = "music2vec-data"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "transformers==4.44.2",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "numpy<2",
        "nnAudio==0.3.3",
        "huggingface_hub<0.27",
        "muq==0.1.0",            # OpenMuQ official package
    )
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

GPU_TYPE = "A10G"  # ~$1.10/hr; 24 GB; sufficient for 330M-param transformer
TIMEOUT_S = 1800   # 30 min hard cap per call (corpus stays under <5 min)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": vol},
    timeout=TIMEOUT_S,
)
def embed_corpus(model_id: str, work_ids: list[str],
                 audio_subdir: str = "audio") -> dict[str, list[float]]:
    """Embed each work_id with `model_id`. Audio is read from the
    Modal volume at /data/<audio_subdir>/<work_id>/w{0,1,2}.wav.

    Two encoder families are supported:
      • OpenMuQ/MuQ-*  — uses the `muq` PyPI package (no preprocessor)
      • everything else (e.g. m-a-p/MERT-v1-*) — Wav2Vec2 extractor +
        AutoModel via transformers

    Returns {work_id: hidden_dim-vector mean-of-windows vector as list}.
    """
    import os
    import sys
    import time

    import numpy as np
    import soundfile as sf
    import torch

    target_sr = 24_000
    print(f"[modal] loading {model_id} ...", file=sys.stderr, flush=True)
    t0 = time.time()

    is_muq = model_id.startswith("OpenMuQ/MuQ-large") or "/MuQ-" in model_id
    if is_muq:
        from muq import MuQ
        model = MuQ.from_pretrained(model_id).cuda().eval()
        extractor = None
    else:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True).cuda().eval()
    print(f"[modal] model loaded in {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)

    out: dict[str, list[float]] = {}
    n_clips = 0
    t_inf = 0.0

    with torch.inference_mode():
        for wid in work_ids:
            window_vecs = []
            for i in range(3):
                p = f"/data/{audio_subdir}/{wid}/w{i}.wav"
                if not os.path.exists(p):
                    continue
                wav, sr = sf.read(p, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != target_sr:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr,
                                            target_sr=target_sr)
                t1 = time.time()
                if is_muq:
                    wav_t = torch.from_numpy(wav).unsqueeze(0).cuda()
                    out_obj = model(wav_t, output_hidden_states=False)
                    h = (out_obj.last_hidden_state.mean(dim=1)
                         .squeeze(0).cpu().numpy())
                else:
                    inputs = extractor(wav, sampling_rate=target_sr,
                                        return_tensors="pt")
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    h = (model(**inputs)
                         .last_hidden_state.mean(dim=1)
                         .squeeze(0).cpu().numpy())
                t_inf += time.time() - t1
                window_vecs.append(h)
                n_clips += 1
            if window_vecs:
                out[wid] = np.mean(window_vecs, axis=0).astype(
                    "float32").tolist()

    print(f"[modal] embedded {len(out)}/{len(work_ids)} works "
          f"({n_clips} clips, {t_inf:.1f}s GPU inference)",
          file=sys.stderr, flush=True)
    return out


@app.local_entrypoint()
def main(model_id: str = "m-a-p/MERT-v1-330M",
         work_ids_file: str = "data/embeddings/audio_mert95.work_ids.txt"):
    """Local entrypoint — embed the corpus listed in `work_ids_file`.

    Run via:  modal run modal_embed.py --model-id m-a-p/MERT-v1-330M

    For the production driver use scripts/03_modal_embed.py which
    saves results to data/embeddings/ next to the local-CPU baseline.
    """
    from pathlib import Path
    work_ids = Path(work_ids_file).read_text().splitlines()
    print(f"submitting {len(work_ids)} work_ids to Modal "
          f"(model={model_id}) ...")
    result = embed_corpus.remote(model_id, work_ids)
    print(f"got {len(result)} embeddings back")
