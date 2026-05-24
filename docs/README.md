# music2vec — site (`/docs`)

This directory is the source for the project's GitHub Pages site: an
interactive 3D atlas of 179 public-domain musical works placed in the
embedding space of three frozen audio encoders (MERT-95M, MERT-330M,
MuQ-large). Click a sphere to listen.

## Live

GitHub Pages serves `/docs` via `.github/workflows/pages.yml`. Every push
to `main` that touches `docs/` re-deploys.

URL: `https://<owner>.github.io/music2vec/` (resolves once Pages is enabled).

## What lives here

```
docs/
  index.html              hero + canvas + side panels + tour + audio player
  css/style.css           dark, layered glass aesthetic
  js/app.js               three.js scene, raycasting, custom audio player
  data/works.json         179-work dataset + 3D coords (~281 KB)
  audio/<id>.mp3          per-work 30 s preview, 96 kbps mono (~62 MB total)
```

## Regenerating the data

The JSON and audio MP3s are committed so Pages serves a pure static build
with no build step. To rebuild after upstream embeddings or audio change:

```bash
python3 scripts/20_build_site_data.py
```

The script:
1. Loads each of the three audio embeddings (`data/embeddings/audio_*.npy`)
   and orders them by their `work_ids.txt`.
2. Computes a 3D PCA, t-SNE (perplexity 30, cosine), and UMAP (n=15,
   min_dist=0.05) per encoder.
3. Pulls 2D PCA / t-SNE / UMAP from the existing per-encoder parquets for
   the flat-map fallback.
4. Joins six taxonomies (compositional_device, dance_type, instrumentation,
   national_school, opus_cycle, sacred_function) onto each work.
5. Transcodes each work's middle 30 s window (`data/audio/<id>/w1.wav`) to
   96 kbps mono mp3 at 24 kHz via ffmpeg. Idempotent — skips existing.
6. Writes the JSON.

Dependencies: numpy, pandas, scikit-learn, umap-learn, ffmpeg in PATH.

## License

MIT. Audio renders are derived from Mutopia Project LilyPond scores
(public domain) via FluidSynth + the FluidR3 General-MIDI SoundFont.
