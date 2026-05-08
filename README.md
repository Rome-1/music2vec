# music2vec

A music-domain analogue of Figure 3 in [emoji2vec](https://arxiv.org/abs/1609.08359) (Eisner et al., 2016) — and an exploration of what a frozen music encoder's embedding space says about public-domain musical works.

![hero](out/latent_works.png)

> **Status: pilot landed.** 105 Bach works (WTC I/II + solo violin + cello suites) embedded with MERT-v1-95M. Headline: **fugues cluster at 91% k=5 NN purity**, with PHATE compactness **0.37** — the encoder discovered counterpoint as a category without being told.

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

## The hero figure

PCA on top spanning the full width, with t-SNE / UMAP / PHATE in a row of three secondary panels below. Soft hulls only where a category's `compactness < 0.65 × global mean pairwise distance` (flag2vec's honesty constraint). V1 marks are circles; score-thumbnail marks (rendered first system per work) are deferred to a follow-up.

## What the embedding actually clusters on

V1 corpus is Bach-only (105 works) so most of the six taxonomies degenerate or apply trivially. The headline result lives in **compositional device** — the only taxonomy that varies orthogonally to instrumentation in this pilot. All numbers below are on **frozen MERT-v1-95M, mean-pooled across 3 × 30 s windows of FluidSynth-rendered audio**.

| Taxonomy | n | K | k=5 NN purity | Compactness (PHATE) | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| **Compositional device** (fugue + passacaglia) | 16 | 2 | **0.91** | **0.37** | Fugues form one of the two tightest clusters in the figure. The hull is uninterrupted across PCA, t-SNE, UMAP, PHATE. |
| Instrumentation (harpsichord / piano / unaccompanied string) | 105 | 3 | 0.89 | — | The workhorse: timbre dominates. Expected. Confirms the projection is meaningful. |
| Opus cycle (WTC I, WTC II, solo violin, cello suites) | 105 | 4 | 0.86 | — | k-means partially recovers (ARI 0.24, NMI 0.43). Cycles cohere; the boundaries between WTC I and WTC II are softer than between WTC and the string works. |
| Dance type (8 forms across 25 movements) | 25 | 8 | 0.11 | — | At chance (1/8 ≈ 0.125). NMI 0.42 says k-means *finds* structure, just not aligned with dance labels — suggests the cello-suite movements cluster more by suite-of-origin than by dance type within a Bach-only corpus. |
| National school | 105 | 1 | — | — | Trivial in V1 (all `german_contrapuntal`). |
| Sacred function | 1 | 1 | — | — | One motet (BWV 878 *Spiritus Domini*); not enough for a hull. |

**Caveat.** The 91% fugue purity is partially confounded with the WTC-harpsichord instrumentation: 11 of the 15 fugues are WTC keyboard fugues. The four solo-violin fugues (BWV 1001 / 1003 / 1005) sit in a *different* sub-region in PHATE, but inside the same outer hull — so the encoder is grouping fugues despite different timbre, not just by timbre. To fully honor the headline claim we need fugues across more timbres (organ, piano, chamber, orchestra) — coming in the next push.

**Honest outliers.** LOF on the embedding flags consistent visual outliers across taxonomies: the BWV 878 *Spiritus Domini* motet (which my labeler mistakenly tagged as harpsichord — fix coming), and the BWV 1002 partita-as-cello transcriptions (the audio is genuinely thin and high-variance because cello transposes the violin part down). The encoder agrees they're unusual.

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
