# music2vec

A music-domain analogue of Figure 3 in [emoji2vec](https://arxiv.org/abs/1609.08359) (Eisner et al., 2016) — and an exploration of what a frozen music encoder's embedding space says about public-domain musical works.

> 🚧 **Status:** scaffold / pre-data. The pipeline runs end-to-end on a synthetic 5-work fixture; corpus acquisition is the next milestone.

emoji2vec's Figure 3 projects 1,661 emoji embeddings to 2D with t-SNE and renders each emoji glyph at its position. Clusters of similar emojis (smileys, animals, fruits, flags) emerge from a *language*-derived embedding space — and notably, in the original paper, all the country flags pile into one undifferentiated cluster. The flags-cluster inspired sister project [flag2vec](https://github.com/Rome-1/flag2vec); this project asks the parallel question of music.

> Do *audio* embeddings of public-domain musical works recover the *compositional / formal / cultural* groupings a music historian would draw by hand — fugues across 250 years, dance forms by metric signature, sacred function across centuries, opus cycles as coherent blobs?

The bet: a frozen, off-the-shelf music encoder will recover **compositional grammar** (fugue, canon, passacaglia) more cleanly than era or composer alone, because the encoder sees texture statistics — overlapping mid-band energy, periodic bass repetitions — that survive harmonic re-clothing.

## Method

- **Source corpus:** Public-domain MIDI from [Mutopia](https://www.mutopiaproject.org/) (curated, fully PD) plus curated [kunstderfuge](https://www.kunstderfuge.com/) top-up where Mutopia is thin. Target ~1,500 works covering all six taxonomies; pilot at 200.
- **Audio rendering:** FluidSynth + MuseScore General SoundFont, 24 kHz mono. Three 30 s windows per work (start / middle / random-seeded), mean-pooled to wash out window-position artifacts. Optional dual-SoundFont averaging to attenuate synth-identity timbre.
- **Audio encoder (primary):** [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M) — frozen, self-supervised on ~160 k hours of music, the closest philosophical analogue to flag2vec's DINOv2. The 95 M variant (vs 330 M) is a CPU-pragmatic choice on this hardware; the 330 M can be retrofitted on a GPU later.
- **Symbolic encoder (comparison):** [`musicbert`](https://huggingface.co/microsoft/musicbert) on raw MIDI tokens — sharpens contrapuntal-device clustering where audio embeddings smear with timbre.
- **Projections:** **PCA (hero)**, t-SNE (perplexity 30, cosine), UMAP (n_neighbors 15, cosine), PHATE (knn 15, decay 20).
- **Annotation:** six hand-curated taxonomies in `music2vec/taxonomies.py`.

## The hero figure (planned layout)

PCA on top spanning the full width, with t-SNE / UMAP / PHATE in a row of three secondary panels below. Soft hulls only where a category's `compactness < 0.65 × global mean pairwise distance` (flag2vec's honesty constraint). Each work is rendered at its 2D position as a thumbnail of the score's first system; works with no available score render as a composer monogram.

## The six taxonomies

These are the music analogue of flag2vec's vex categories: hand-curated, hypothesis-driven, drawn as soft hulls.

1. **Compositional device** *(headline)* — fugue, canon, passacaglia/chaconne, theme & variations, ostinato, twelve-tone, isorhythm, free imitation.
   *Hypothesis:* fugues cluster across 250 years of harmonic vocabulary; the encoder learned grammar, not vocabulary.
2. **Dance type** — minuet, waltz, mazurka, polonaise, sarabande, gigue, tarantella, allemande, courante, gavotte, bourrée, habanera, siciliana, march.
   *Hypothesis:* the Baroque suite (allemande → courante → sarabande → gigue) appears as a tight ring of adjacent hulls.
3. **Instrumentation–texture** — solo keyboard (piano / harpsichord / organ), unaccompanied string, string quartet, piano trio, lieder, SATB chorus, full orchestra, opera aria, wind ensemble, lute/guitar.
   *Probe:* WTC-on-harpsichord vs WTC-on-piano — does the encoder cluster on content or container?
4. **Sacred function** — chant, Mass Ordinary / Proper, requiem, motet, Lutheran chorale, chorale prelude, Passion, oratorio, Magnificat, Stabat Mater, Te Deum, anthem, psalm setting.
   *Probe:* do 20c spiritual minimalists (Pärt / Tavener / Górecki) cluster with Renaissance polyphony, skipping 300 years?
5. **Opus cycle** — Bach WTC I/II, Goldberg, Art of Fugue, cello suites, solo violin; Beethoven 32; Chopin Op. 28 / 10 / 25; Scriabin Op. 11; Shostakovich Op. 87; Debussy Préludes I/II; Liszt Transcendentals; Bartók Mikrokosmos; Ligeti Études.
   *Probe:* WTC I and II overlap nearly perfectly (Bach was consistent across 22 y); Beethoven 32 splits into early/middle/late sub-hulls.
6. **National school** — German contrapuntal, French clavecinistes, Italian operatic, Mighty Handful, Czech nationalist, Iberian, Nordic, English pastoral, Second Viennese, Les Six, American vernacular, Hungarian folk-based.
   *Probe:* which "schools" the encoder refuses to separate (the pan-Arab-vs-pan-African energy: shared sonic vocabulary across nominal boundaries).

Era is intentionally **not** a hull — it goes underneath as a background gradient (medieval/renaissance, baroque, classical, romantic, modern/contemporary).

## Pipeline

| Script | Purpose |
| --- | --- |
| `scripts/01_acquire.py` | Pull Mutopia + kunstderfuge MIDI; write `data/works.csv` |
| `scripts/02_render.py` | FluidSynth → 24 kHz WAV; 3 × 30 s window crops per work |
| `scripts/03_embed_audio.py` | MERT-v1-95M frozen embedding, mean over windows + time |
| `scripts/03c_embed_symbolic.py` | MusicBERT-on-MIDI symbolic embedding (comparison panel) |
| `scripts/04_project.py` | PCA / t-SNE / UMAP / PHATE → `data/projections/*.parquet` |
| `scripts/05_render.py` | Hero figure: PCA on top, t-SNE / UMAP / PHATE row below |
| `scripts/06_per_projection.py` | Single-projection large figures (one per method) — *next* |
| `scripts/07_per_category.py` | Per-taxonomy highlight figures with soft hulls — *next* |
| `scripts/09_clustering.py` | k-NN purity, k-means confusion, linear probe, LOF, dendrogram — *next* |

## Compute envelope

This box has no GPU, 4 CPU cores, ~10 GiB RAM available. Choices reflect that: MERT-95M (not 330M), pilot-first corpus, resumable scripts, checkpoint-friendly intermediates. Full 1,500-work pipeline is estimated at ~3-4 hours wall on this hardware end-to-end.

## Stack

Python 3.10+, PyTorch (CPU), Hugging Face Transformers, scikit-learn, umap-learn, phate, FluidSynth (system binary + `pyfluidsynth`), `mido` / `pretty_midi` / `music21` for symbolic, soundfile + librosa for audio I/O, matplotlib + Pillow for figures.

```bash
pip install -r requirements.txt
# system: apt install fluidsynth fluid-soundfont-gm  (or equivalent)
```

## Layout

```
music2vec/
├── music2vec/                 # shared library
│   ├── style.py               # palette, hull helpers, thumbnail loader
│   └── taxonomies.py          # six taxonomies + era bins
├── data/
│   ├── works.csv              # canonical work list (tracked)
│   ├── taxonomies/            # per-taxonomy work_id→label CSVs (tracked)
│   ├── raw/                   # MIDI sources (gitignored)
│   ├── audio/                 # rendered WAV (gitignored)
│   ├── embeddings/            # .npy (gitignored)
│   └── projections/           # small parquet (tracked)
├── scripts/                   # numbered pipeline
└── out/                       # rendered figures (tracked)
```

## License

Code: MIT. Source MIDI/scores remain under their original public-domain status; per-source provenance and verification recorded in `data/works.csv`.

## Acknowledgements

Sister project [flag2vec](https://github.com/Rome-1/flag2vec) for the structural template, the aesthetic, and the hull-honesty constraint. emoji2vec (Eisner et al., 2016) for Figure 3.
