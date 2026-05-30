"""Taxonomies for music2vec — the "vex categories" of the music domain.

Six hand-curated taxonomies that we expect a frozen music encoder to
*partially* recover. Each taxonomy is a flat label space; a single work
may carry one label per taxonomy (or none). Per-work assignments live in
`data/taxonomies/<taxonomy>.csv` with columns (work_id, label).

Per Rome (2026-05-08): all six are in scope; lead with compositional
device for the headline figure, but pivot if results say otherwise.
Era is intentionally NOT a hull — it goes underneath as a background
gradient.
"""
from __future__ import annotations

# ---------- 1. Compositional device / contrapuntal texture ----------
# The headline taxonomy. Hypothesis: a frozen audio encoder picks up
# contrapuntal density and thematic-repetition structure, so fugues
# cluster across 250 years of harmonic vocabulary.
COMPOSITIONAL_DEVICE = {
    "fugue":              "#2E5DA5",
    "canon":              "#3F73B8",
    "passacaglia":        "#5A4E8F",   # incl. chaconne, ground-bass
    "theme_variations":   "#7E6BA3",
    "ostinato":           "#9C8A5C",
    "twelve_tone":        "#3D3D3D",
    "isorhythm":          "#7E5A1F",
    "free_imitation":     "#6E8C73",   # stretti / fugato episodes
}

# ---------- 2. Dance-type / metric-choreographic identity ----------
# Hypothesis: rhythmic surface signatures (mazurka beat-2, polonaise
# bass figure, sarabande accent on 2) are the first thing audio
# encoders latch onto.
DANCE_TYPE = {
    "minuet":      "#A04C2C",
    "waltz":       "#C0892F",
    "mazurka":     "#8C5A2E",
    "polonaise":   "#7E2F2F",
    "sarabande":   "#5C4A3A",
    "gigue":       "#B26A3A",
    "tarantella":  "#D4843A",
    "allemande":   "#6E4A3A",
    "courante":    "#8E6A4A",
    "gavotte":     "#A07E5A",
    "bourree":     "#9C7050",
    "habanera":    "#8C3A4A",
    "siciliana":   "#6E5C9C",
    "march":       "#3D3D3D",
}

# ---------- 3. Instrumentation-texture ----------
# The "sanity check" — audio encoders will absolutely cluster these.
# Probes: WTC-on-harpsichord vs WTC-on-piano (content vs container).
INSTRUMENTATION = {
    "solo_keyboard_piano":      "#3E8C73",
    "solo_keyboard_harpsichord":"#5AA08A",
    "solo_keyboard_organ":      "#1F7A6B",
    "unaccompanied_string":     "#7E9C8A",   # solo violin/cello suites
    "string_quartet":           "#6BA08A",
    "piano_trio":               "#9CB8A8",
    "lieder_voice_piano":       "#C77A3E",   # voice + keyboard
    "satb_a_cappella":          "#A0A0A0",
    "mass_with_orchestra":      "#6E5C9C",
    "opera_aria":               "#A02C2C",
    "full_orchestra":           "#1F3F6E",
    "wind_ensemble":            "#7A8C3E",
    "lute_guitar_solo":         "#8C7050",
}

# ---------- 4. Sacred function / liturgical genre ----------
# Co-evolved acoustic conventions + modal/diatonic restraint + reverberant
# capture. Probe: do 20c spiritual minimalists (Pärt/Tavener/Górecki)
# cluster with Renaissance polyphony, skipping 300 years?
SACRED_FUNCTION = {
    "gregorian_chant":     "#A0A0A0",
    "mass_ordinary":       "#6E5C9C",
    "mass_proper":         "#8E7CB8",
    "requiem":             "#3D3D3D",
    "motet":               "#5C7C46",
    "lutheran_chorale":    "#7E5A1F",
    "chorale_prelude":     "#9C8A5C",
    "oratorio":            "#A02C2C",
    "passion":             "#7E2F2F",
    "magnificat":          "#3E8C73",
    "stabat_mater":        "#6E4A3A",
    "te_deum":             "#1F3F6E",
    "anthem":              "#C77A3E",
    "psalm_setting":       "#5AA08A",
}

# ---------- 5. Sibling-set / opus-cycle membership ----------
# Probes intra-cycle coherence. Bet: WTC I and II overlap nearly perfectly
# (Bach was consistent across 22y); Beethoven 32 splits into early/middle/
# late sub-hulls.
OPUS_CYCLE = {
    "bach_wtc_1":               "#2E5DA5",
    "bach_wtc_2":               "#3F73B8",
    "bach_goldberg":            "#5A4E8F",
    "bach_art_of_fugue":        "#3D3D3D",
    "bach_cello_suites":        "#7E5A1F",
    "bach_solo_violin":         "#9C8A5C",
    "beethoven_32_sonatas":     "#A02C2C",
    "chopin_op28_preludes":     "#C77A3E",
    "chopin_op10_etudes":       "#A04C2C",
    "chopin_op25_etudes":       "#8C5A2E",
    "scriabin_op11_preludes":   "#6E5C9C",
    "shostakovich_op87":        "#1F3F6E",
    "debussy_preludes_1":       "#3E8C73",
    "debussy_preludes_2":       "#5AA08A",
    "liszt_transcendentals":    "#7E2F2F",
    "bartok_mikrokosmos":       "#7A8C3E",
    "ligeti_etudes":            "#3D3D3D",
}

# ---------- 6. National school ----------
# Reveals which "schools" the encoder refuses to separate (pan-Arab vs
# pan-African energy: shared sonic vocabulary across nominal boundaries).
NATIONAL_SCHOOL = {
    "german_contrapuntal":   "#1F3F6E",
    "french_clavecinistes":  "#6E5C9C",
    "italian_operatic":      "#A02C2C",
    "polish_romantic":       "#A04C2C",   # Chopin and successors
    "mighty_handful":        "#7E2F2F",
    "czech_nationalist":     "#5C7C46",
    "iberian":               "#C0892F",
    "nordic":                "#3E8C73",
    "english_pastoral":      "#7A8C3E",
    "second_viennese":       "#3D3D3D",
    "les_six":               "#8E6A4A",
    "american_vernacular":   "#3F73B8",
    "hungarian_folk_based":  "#7E5A1F",   # Bartók/Kodály/Liszt rhapsodies
}


# Composer palette — used by single-projection figures and the hero
# cross-projection comparison to color works by their author. Distinct
# print-safe hues, no two adjacent on a colorblind ramp.
COMPOSER_COLORS = {
    "BachJS":     "#1F3F6E",   # deep cobalt — anchor of the corpus
    "BeethovenLv":"#A04C2C",   # terracotta — bridges baroque + romantic
    "ChopinFF":   "#A02C58",   # rose — Polish-Romantic island
    "DvorakA":    "#3E8C73",   # forest — Czech nationalist
    "MozartWA":   "#C0892F",   # ochre — Viennese classical
    "VivaldiA":   "#6E5C9C",   # plum — Italian baroque concerto
    "HandelGF":   "#7E5A1F",   # umber — German-baroque-via-London
}

COMPOSER_NICE_NAME = {
    "BachJS":      "Bach",
    "BeethovenLv": "Beethoven",
    "ChopinFF":    "Chopin",
    "DvorakA":     "Dvořák",
    "MozartWA":    "Mozart",
    "VivaldiA":    "Vivaldi",
    "HandelGF":    "Handel",
}


# Registry of all taxonomies. Used by the figure pipeline to iterate.
TAXONOMIES = {
    "compositional_device": COMPOSITIONAL_DEVICE,
    "dance_type":           DANCE_TYPE,
    "instrumentation":      INSTRUMENTATION,
    "sacred_function":      SACRED_FUNCTION,
    "opus_cycle":           OPUS_CYCLE,
    "national_school":      NATIONAL_SCHOOL,
}

NICE_NAME = {
    "compositional_device": "Compositional device",
    "dance_type":           "Dance type",
    "instrumentation":      "Instrumentation",
    "sacred_function":      "Sacred function",
    "opus_cycle":           "Opus cycle",
    "national_school":      "National school",
}


# Era is the background gradient, not a hull. Five bins, anchored to
# canonical period boundaries; works falling outside ~1400-2025 are
# clamped to the endpoints.
ERA_BINS = [
    ("medieval_renaissance", 1400, 1600, "#7E5A1F"),
    ("baroque",              1600, 1750, "#A04C2C"),
    ("classical",            1750, 1820, "#C0892F"),
    ("romantic",             1820, 1910, "#3E8C73"),
    ("modern_contemporary",  1910, 2025, "#1F3F6E"),
]
