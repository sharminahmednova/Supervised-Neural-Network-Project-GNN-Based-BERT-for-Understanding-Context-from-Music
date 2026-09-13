"""Procedural music corpus -- a stand-in for FMA / MagnaTagATune / MusicCaps.

None of the real corpora ship with this repository (FMA alone is ~20 GB), so
this module *composes* audio from genre grammars: each genre has its own tempo
range, chord vocabulary, harmonic profile, percussion density and brightness,
and each track additionally carries a mood that shifts its mode and dynamics.

The point is that the three modalities are genuinely coupled:

    genre + mood  ->  audio (chords, timbre, tempo)
                  ->  tags (multi-label)
                  ->  caption / lyric text (MusicCaps-style)
                  ->  valence & arousal (DEAM-style, continuous)

so a text-only model, a graph-only model and a fused model all find real signal,
and their gaps mean something. Swap `dataset.source` in config.yaml to train the
identical code on the real archives; see `adapters.py`.

Every track is assigned a synthetic `artist`, so the grouped splitter can be
exercised against artist leakage exactly as it would be on FMA.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 22050

# --------------------------------------------------------------------------- #
# Genre grammar
# --------------------------------------------------------------------------- #
GENRES = ["jazz", "rock", "classical", "electronic", "hiphop", "folk", "metal", "ambient"]

# root offsets (semitones from C) for the chord loop of each genre
GENRE_SPEC = {
    #                progression (semitone roots),  qualities,      tempo,     brightness, perc
    "jazz":       dict(roots=[0, 5, 7, 2],  quals=["min", "maj", "maj", "min"], tempo=(90, 140),  bright=0.55, perc=0.35),
    "rock":       dict(roots=[0, 7, 9, 5],  quals=["maj", "maj", "min", "maj"], tempo=(110, 150), bright=0.70, perc=0.75),
    "classical":  dict(roots=[0, 5, 7, 0],  quals=["maj", "maj", "maj", "maj"], tempo=(60, 100),  bright=0.45, perc=0.05),
    "electronic": dict(roots=[9, 5, 0, 7],  quals=["min", "maj", "maj", "maj"], tempo=(120, 140), bright=0.85, perc=0.90),
    "hiphop":     dict(roots=[9, 2, 7, 9],  quals=["min", "min", "maj", "min"], tempo=(80, 100),  bright=0.50, perc=0.95),
    "folk":       dict(roots=[0, 5, 9, 7],  quals=["maj", "maj", "min", "maj"], tempo=(90, 120),  bright=0.60, perc=0.25),
    "metal":      dict(roots=[4, 0, 5, 7],  quals=["min", "min", "maj", "min"], tempo=(140, 190), bright=0.95, perc=0.85),
    "ambient":    dict(roots=[0, 9, 5, 7],  quals=["maj", "min", "maj", "maj"], tempo=(50, 75),   bright=0.30, perc=0.02),
}

GENRE_INSTRUMENTS = {
    "jazz": ["saxophone", "piano", "drums"],
    "rock": ["guitar", "drums", "bass"],
    "classical": ["strings", "piano"],
    "electronic": ["synth", "drums"],
    "hiphop": ["drums", "bass", "synth"],
    "folk": ["guitar", "vocals"],
    "metal": ["guitar", "drums", "bass"],
    "ambient": ["synth", "strings"],
}

# mood -> (valence, arousal) centre on the DEAM 1-9 scale
MOODS = {
    "melancholic": (3.0, 3.0),
    "calm":        (6.0, 2.5),
    "happy":       (7.5, 6.5),
    "energetic":   (6.5, 8.0),
    "aggressive":  (3.0, 8.5),
    "dreamy":      (6.0, 3.5),
}

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_QUAL_INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7)}


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _adsr(n: int, sr: int, attack=0.02, release=0.25) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    a = min(int(attack * sr), n // 2)
    r = min(int(release * sr), n // 2)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def _chord_wave(root: int, qual: str, n: int, sr: int, brightness: float,
                rng: np.random.Generator) -> np.ndarray:
    """One chord: triad partials with a brightness-controlled harmonic series."""
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / sr
    octave = 48 + 12 * int(rng.integers(0, 2))  # C3 or C4 register
    for step in _QUAL_INTERVALS[qual]:
        f0 = _midi_to_hz(octave + root + step)
        # more partials (and slower rolloff) = brighter timbre
        n_partials = 2 + int(brightness * 7)
        for h in range(1, n_partials + 1):
            if f0 * h > sr / 2.2:
                break
            amp = (brightness ** 0.5) / (h ** (2.2 - brightness))
            phase = float(rng.uniform(0, 2 * np.pi))
            y += amp * np.sin(2 * np.pi * f0 * h * t + phase)
    return y * _adsr(n, sr)


def _percussion(n: int, sr: int, tempo: float, density: float,
                rng: np.random.Generator) -> np.ndarray:
    """Noise-burst kick/hat grid at the track tempo."""
    y = np.zeros(n, dtype=np.float32)
    if density <= 0.0:
        return y
    step = max(int(sr * 60.0 / tempo / 2.0), 1)   # eighth notes
    for k, start in enumerate(range(0, n, step)):
        if rng.random() > density:
            continue
        is_kick = k % 4 == 0
        length = min(int(sr * (0.12 if is_kick else 0.04)), n - start)
        if length <= 0:
            continue
        noise = rng.standard_normal(length).astype(np.float32)
        decay = np.exp(-np.linspace(0, 6.0 if is_kick else 14.0, length)).astype(np.float32)
        if is_kick:  # low-pass the kick by cumulative smoothing
            noise = np.convolve(noise, np.ones(24, dtype=np.float32) / 24, mode="same")
            noise += 0.8 * np.sin(2 * np.pi * 55.0 * np.arange(length) / sr)
        y[start:start + length] += 0.55 * noise * decay
    return y


def synthesize_track(genre: str, mood: str, seed: int, duration: float = 30.0,
                     sr: int = SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    """Compose one track. Returns (waveform, ground-truth attribute dict)."""
    rng = np.random.default_rng(seed)
    spec = GENRE_SPEC[genre]
    tempo = float(rng.uniform(*spec["tempo"]))
    key = int(rng.integers(0, 12))

    # Mood nudges brightness and dynamics away from the genre centre.
    valence, arousal = MOODS[mood]
    brightness = float(np.clip(spec["bright"] + (arousal - 5.0) * 0.04, 0.15, 0.98))
    perc_density = float(np.clip(spec["perc"] + (arousal - 5.0) * 0.05, 0.0, 1.0))

    bar_seconds = 4 * 60.0 / tempo
    n_total = int(duration * sr)
    n_bar = max(int(bar_seconds * sr), sr // 2)

    y = np.zeros(n_total, dtype=np.float32)
    chord_sequence = []
    for bar_i, start in enumerate(range(0, n_total, n_bar)):
        length = min(n_bar, n_total - start)
        if length < sr // 10:
            break
        slot = bar_i % len(spec["roots"])
        root = (key + spec["roots"][slot]) % 12
        qual = spec["quals"][slot]
        # minor-mode pull for low-valence moods
        if valence < 4.5 and qual == "maj" and rng.random() < 0.35:
            qual = "min"
        y[start:start + length] += _chord_wave(root, qual, length, sr, brightness, rng)
        chord_sequence.append(f"{PITCH_CLASSES[root]}:{qual}")

    y += _percussion(n_total, sr, tempo, perc_density, rng)
    y += 0.01 * rng.standard_normal(n_total).astype(np.float32)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = (y / peak * 0.9).astype(np.float32)

    meta = {
        "genre": genre,
        "mood": mood,
        "tempo": round(tempo, 1),
        "key": PITCH_CLASSES[key],
        "brightness": round(brightness, 3),
        "instruments": GENRE_INSTRUMENTS[genre],
        "chord_sequence": chord_sequence,
        "valence": float(np.clip(valence + rng.normal(0, 0.45), 1.0, 9.0)),
        "arousal": float(np.clip(arousal + rng.normal(0, 0.45), 1.0, 9.0)),
    }
    return y, meta


# --------------------------------------------------------------------------- #
# Text + tags
# --------------------------------------------------------------------------- #
def _tempo_word(tempo: float) -> str:
    if tempo < 75:
        return "slow"
    if tempo < 110:
        return "mid-tempo"
    return "fast"


def build_tags(meta: dict) -> list[str]:
    """Multi-label tag set: genre + mood + instruments + tempo/texture tags."""
    tags = [meta["genre"], meta["mood"]] + list(meta["instruments"])
    tags.append(_tempo_word(meta["tempo"]))
    tags.append("bright" if meta["brightness"] > 0.6 else "dark")
    if meta["arousal"] > 6.5:
        tags.append("loud")
    if meta["arousal"] < 3.5:
        tags.append("quiet")
    if any(c.endswith(":min") for c in meta["chord_sequence"]):
        tags.append("minor key")
    else:
        tags.append("major key")
    if meta["valence"] > 6.0:
        tags.append("upbeat")
    if meta["valence"] < 4.0:
        tags.append("sad")
    return sorted(set(tags))


_CAPTION_FRAMES = [
    "A {tempo} {genre} piece with a {mood} feel, driven by {instruments}.",
    "This is a {mood} {genre} recording in {key}. {instruments_cap} carry the arrangement at around {bpm} BPM.",
    "{tempo_cap} {genre} track. The harmony moves through {progression}, and the overall mood is {mood}.",
    "A {mood}, {texture} {genre} instrumental featuring {instruments}, roughly {bpm} beats per minute.",
]

_LYRIC_FRAMES = [
    "the night keeps turning and the {mood} light stays on",
    "we were {mood} in the {key} of every quiet room",
    "hold the line, {tempo} hearts under a {texture} sky",
    "nothing louder than a {mood} song at {bpm} beats",
]


# Vaguer stand-ins used when a caption redacts its literal label word. Real
# MusicCaps captions describe a clip without reliably naming its tag set, so
# without this the caption would *be* the label and BERT-only would score ~1.0,
# leaving nothing for the graph branch to add. `label_dropout` controls how
# often the giveaway word is swapped for a generic one.
_GENRE_HEDGE = ["music", "instrumental", "studio", "band"]
_MOOD_HEDGE = ["distinctive", "particular", "certain", "unmistakable"]


def build_caption(meta: dict, rng: np.random.Generator, label_dropout: float = 0.0) -> str:
    """MusicCaps-style free-text description."""
    instruments = ", ".join(meta["instruments"])
    progression = " to ".join(dict.fromkeys(meta["chord_sequence"][:4])) or "a static drone"
    frame = _CAPTION_FRAMES[int(rng.integers(0, len(_CAPTION_FRAMES)))]

    genre = meta["genre"]
    mood = meta["mood"]
    if label_dropout > 0.0:
        if rng.random() < label_dropout:
            genre = _GENRE_HEDGE[int(rng.integers(0, len(_GENRE_HEDGE)))]
        if rng.random() < label_dropout:
            mood = _MOOD_HEDGE[int(rng.integers(0, len(_MOOD_HEDGE)))]

    return frame.format(
        tempo=_tempo_word(meta["tempo"]),
        tempo_cap=_tempo_word(meta["tempo"]).capitalize(),
        genre=genre,
        mood=mood,
        key=meta["key"],
        bpm=int(round(meta["tempo"])),
        instruments=instruments,
        instruments_cap=instruments.capitalize(),
        progression=progression,
        texture="bright" if meta["brightness"] > 0.6 else "dark",
    )


def build_lyrics(meta: dict, rng: np.random.Generator, label_dropout: float = 0.0) -> str:
    mood = meta["mood"]
    if label_dropout > 0.0 and rng.random() < label_dropout:
        mood = _MOOD_HEDGE[int(rng.integers(0, len(_MOOD_HEDGE)))]
    frame = _LYRIC_FRAMES[int(rng.integers(0, len(_LYRIC_FRAMES)))]
    return frame.format(
        mood=mood,
        key=meta["key"],
        tempo=_tempo_word(meta["tempo"]),
        texture="bright" if meta["brightness"] > 0.6 else "dark",
        bpm=int(round(meta["tempo"])),
    )


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def generate_corpus(n_tracks: int = 300, duration: float = 30.0, seed: int = 42,
                    sr: int = SAMPLE_RATE, n_artists: int | None = None,
                    label_dropout: float = 0.5):
    """Yield (waveform, record) pairs for `n_tracks` procedurally composed tracks.

    Tracks are grouped under synthetic artists so that grouped splitting has
    something to hold out -- an artist stays inside one genre, which is exactly
    the leakage pattern the spec warns about on FMA.
    """
    rng = np.random.default_rng(seed)
    moods = list(MOODS)
    n_artists = n_artists or max(n_tracks // 6, len(GENRES))

    # Each artist keeps one genre and a small mood palette.
    artists = []
    for a in range(n_artists):
        genre = GENRES[a % len(GENRES)]
        palette = list(rng.choice(moods, size=int(rng.integers(1, 3)), replace=False))
        artists.append({"artist_id": f"artist_{a:03d}", "genre": genre, "moods": palette})

    for i in range(n_tracks):
        artist = artists[i % n_artists]
        mood = str(rng.choice(artist["moods"]))
        y, meta = synthesize_track(artist["genre"], mood, seed=seed + i,
                                   duration=duration, sr=sr)
        track_rng = np.random.default_rng(seed + 10_000 + i)
        record = {
            "track_id": f"syn_{i:05d}",
            "artist": artist["artist_id"],
            "genre": meta["genre"],
            "mood": meta["mood"],
            "tags": build_tags(meta),
            "caption": build_caption(meta, track_rng, label_dropout),
            "lyrics": build_lyrics(meta, track_rng, label_dropout),
            "valence": round(meta["valence"], 4),
            "arousal": round(meta["arousal"], 4),
            "tempo": meta["tempo"],
            "key": meta["key"],
            "chord_sequence": meta["chord_sequence"],
        }
        yield y, record
