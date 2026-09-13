"""Readers for the real corpora listed in the spec (Table 1).

Every adapter yields uniform records so the rest of the pipeline never learns
which corpus it is looking at:

    {
      "track_id": str,
      "artist":   str,          # split grouping key -- prevents artist leakage
      "audio_path": str | None, # None means the waveform is supplied directly
      "genre":    str | None,
      "tags":     list[str],
      "caption":  str,          # MusicCaps caption / lyrics / tag-string fallback
      "lyrics":   str,
      "valence":  float | None, # DEAM 1-9
      "arousal":  float | None,
    }

Each adapter documents the directory layout it expects under `data/raw/`.
They are deliberately tolerant: a missing optional file degrades the record
rather than raising, so a partial download still trains.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from audio_features import load_audio


def _tags_to_caption(tags: list[str], genre: str | None = None) -> str:
    """Fallback 'caption' for corpora with tags but no free text."""
    if not tags:
        return f"a {genre} music track" if genre else "a music track"
    head = f"A {genre} track" if genre else "A music track"
    return f"{head} tagged {', '.join(tags)}."


# --------------------------------------------------------------------------- #
# Official splits
#
# Spec section 3, step 5: "Use official FMA / MagnaTagATune splits; for DEAM use
# standard train/val partition". Records carry an optional "split" field; when
# every record has one, dataset.py uses it verbatim instead of resplitting.
# Corpora with no official split (GTZAN) leave it None and fall back to the
# artist-grouped splitter.
# --------------------------------------------------------------------------- #
_SPLIT_ALIASES = {
    "training": "train", "train": "train",
    "validation": "val", "valid": "val", "val": "val",
    "test": "test", "testing": "test",
}


def normalise_split(name: str | None) -> str | None:
    if not name:
        return None
    return _SPLIT_ALIASES.get(str(name).strip().lower())


def magnatagatune_split(mp3_path: str) -> str:
    """MagnaTagATune's standard partition, keyed on the clip's hex directory.

    The community-standard split (used by the published tagging baselines)
    assigns directories 0-b to train, c to validation, d-f to test. Splitting
    this way rather than randomly is what makes a Macro-F1 comparable to a
    number from a paper.
    """
    folder = Path(mp3_path).parent.name.lower()
    if not folder:
        return "train"
    first = folder[0]
    if first in "0123456789ab":
        return "train"
    if first == "c":
        return "val"
    return "test"


# --------------------------------------------------------------------------- #
# GTZAN -- 10 genres, 1000 x 30 s clips
#   data/raw/gtzan/genres_original/<genre>/<genre>.00000.wav
# --------------------------------------------------------------------------- #
def read_gtzan(root: Path) -> list[dict]:
    base = Path(root) / "gtzan"
    for candidate in (base / "genres_original", base / "genres", base):
        if candidate.is_dir():
            base = candidate
            break
    records = []
    for genre_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        genre = genre_dir.name.lower()
        files = sorted(genre_dir.glob("*.wav")) + sorted(genre_dir.glob("*.au"))
        for audio in files:
            # The commonly mirrored tarball was packed on macOS, so every real
            # clip is shadowed by a tiny "._name.wav" AppleDouble resource fork.
            # They are not audio; decoding them yields garbage or an exception.
            if audio.name.startswith("._") or audio.stat().st_size < 10_000:
                continue
            tid = audio.stem
            # GTZAN ships no artist metadata at all. Clips are numbered in
            # recording order, so neighbouring indices are frequently the same
            # artist or even the same track -- a documented flaw in the dataset
            # (Sturm, 2013). Grouping by index *decade* keeps those siblings on
            # one side of the split boundary.
            #
            # Indices are 5-digit zero-padded (00000-00099), so slicing the
            # leading characters collapses every clip in a genre into one group
            # and the splitter then sends whole genres to one split -- train and
            # test end up with disjoint label sets. Parse the number instead.
            digits = "".join(ch for ch in tid.split(".")[-1] if ch.isdigit())
            decade = int(digits) // 10 if digits else 0
            records.append({
                "track_id": tid,
                "artist": f"{genre}_{decade:02d}",
                "audio_path": str(audio),
                "genre": genre,
                "tags": [genre],
                "caption": _tags_to_caption([genre], genre),
                "lyrics": "",
                "valence": None,
                "arousal": None,
                # GTZAN has no official partition; fall through to the
                # artist-grouped splitter.
                "split": None,
            })
    return records


# --------------------------------------------------------------------------- #
# FMA small / medium
#   data/raw/fma/fma_small/<xxx>/<track_id>.mp3
#   data/raw/fma/fma_metadata/tracks.csv
# --------------------------------------------------------------------------- #
def read_fma(root: Path, subset: str = "small") -> list[dict]:
    base = Path(root) / "fma"
    audio_root = base / f"fma_{subset}"
    meta_csv = base / "fma_metadata" / "tracks.csv"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"expected FMA audio at {audio_root}")

    meta: dict[int, dict] = {}
    if meta_csv.exists():
        # tracks.csv has a 3-row hierarchical header; read it positionally.
        with open(meta_csv, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            h0, h1 = next(reader), next(reader)
            next(reader)
            cols = {f"{a}|{b}": i for i, (a, b) in enumerate(zip(h0, h1))}
            genre_i = cols.get("track|genre_top")
            artist_i = cols.get("artist|name")
            title_i = cols.get("track|title")
            tags_i = cols.get("track|tags")
            split_i = cols.get("set|split")        # official FMA partition
            for row in reader:
                if not row or not row[0].strip().isdigit():
                    continue
                tid = int(row[0])
                raw_tags = row[tags_i] if tags_i is not None and tags_i < len(row) else "[]"
                try:
                    tags = [str(t).strip().lower() for t in json.loads(raw_tags.replace("'", '"'))]
                except (json.JSONDecodeError, AttributeError):
                    tags = []
                meta[tid] = {
                    "genre": (row[genre_i].strip().lower()
                              if genre_i is not None and genre_i < len(row) and row[genre_i].strip()
                              else None),
                    "artist": (row[artist_i].strip()
                               if artist_i is not None and artist_i < len(row) else ""),
                    "title": (row[title_i].strip()
                              if title_i is not None and title_i < len(row) else ""),
                    "tags": tags,
                    "split": normalise_split(
                        row[split_i] if split_i is not None and split_i < len(row) else None),
                }

    records = []
    for audio in sorted(audio_root.rglob("*.mp3")):
        try:
            tid = int(audio.stem)
        except ValueError:
            continue
        m = meta.get(tid, {})
        genre = m.get("genre")
        if genre is None:
            continue                      # untagged tracks carry no label to learn
        tags = sorted({genre, *m.get("tags", [])})
        records.append({
            "track_id": f"fma_{tid}",
            "artist": m.get("artist") or f"unknown_{tid}",
            "audio_path": str(audio),
            "genre": genre,
            "tags": tags,
            "caption": _tags_to_caption(tags, genre) if not m.get("title")
                       else f'"{m["title"]}" by {m.get("artist", "unknown")}. {_tags_to_caption(tags, genre)}',
            "lyrics": "",
            "valence": None,
            "arousal": None,
            "split": m.get("split"),
        })
    return records


# --------------------------------------------------------------------------- #
# MagnaTagATune -- 188 tags, 25,877 clips
#   data/raw/magnatagatune/annotations_final.csv   (tab separated)
#   data/raw/magnatagatune/audio/<f>/<clip>.mp3
# --------------------------------------------------------------------------- #
def read_magnatagatune(root: Path, top_tags: int = 50) -> list[dict]:
    base = Path(root) / "magnatagatune"
    ann = base / "annotations_final.csv"
    if not ann.exists():
        raise FileNotFoundError(f"expected MagnaTagATune annotations at {ann}")
    audio_root = base / "audio" if (base / "audio").is_dir() else base

    with open(ann, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    tag_names = [k for k in rows[0] if k not in {"clip_id", "mp3_path"}]

    counts = {t: 0 for t in tag_names}
    for row in rows:
        for t in tag_names:
            if row.get(t) == "1":
                counts[t] += 1
    keep = [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_tags]]

    records = []
    for row in rows:
        rel = (row.get("mp3_path") or "").strip()
        if not rel:
            continue
        tags = sorted(t for t in keep if row.get(t) == "1")
        if not tags:
            continue                      # clips with no top-K tag teach nothing
        path = audio_root / rel
        records.append({
            "track_id": f"mtat_{row['clip_id']}",
            # mp3_path is "<dir>/<artist>-<album>-<track>.mp3" -- the filename
            # stem up to the first dash is the artist, which is the grouping key.
            "artist": Path(rel).stem.split("-")[0] or Path(rel).parent.name,
            "audio_path": str(path),
            "genre": None,
            "tags": tags,
            "caption": _tags_to_caption(tags),
            "lyrics": "",
            "valence": None,
            "arousal": None,
            "split": magnatagatune_split(rel),
        })
    return records


# --------------------------------------------------------------------------- #
# MusicCaps -- 5,521 clips with expert captions
#   data/raw/musiccaps/musiccaps-public.csv
#   data/raw/musiccaps/audio/<ytid>.wav   (downloaded separately)
# --------------------------------------------------------------------------- #
def read_musiccaps(root: Path) -> list[dict]:
    base = Path(root) / "musiccaps"
    csv_path = next((p for p in (base / "musiccaps-public.csv", base / "musiccaps.csv")
                     if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(f"expected musiccaps-public.csv under {base}")
    audio_root = base / "audio"

    records = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ytid = row.get("ytid", "").strip()
            if not ytid:
                continue
            audio = next((p for p in (audio_root / f"{ytid}.wav", audio_root / f"{ytid}.mp3")
                          if p.exists()), None)
            if audio is None:
                continue                  # caption present but clip not downloaded
            aspects = row.get("aspect_list", "[]")
            try:
                tags = [str(a).strip().lower() for a in json.loads(aspects.replace("'", '"'))]
            except (json.JSONDecodeError, AttributeError):
                tags = []
            # MusicCaps marks its held-out evaluation clips with is_audioset_eval;
            # the remainder is the train pool, which we halve into train/val.
            is_eval = str(row.get("is_audioset_eval", "")).strip().lower() in {"true", "1"}
            split = "test" if is_eval else ("val" if len(records) % 8 == 0 else "train")
            records.append({
                "track_id": f"mc_{ytid}",
                "artist": ytid,           # one clip per video: no cross-clip leakage
                "audio_path": str(audio),
                "genre": None,
                "tags": tags,
                "caption": row.get("caption", "").strip(),
                "lyrics": "",
                "valence": None,
                "arousal": None,
                "split": split,
            })
    return records


# --------------------------------------------------------------------------- #
# DEAM -- continuous valence/arousal, merged into an existing record list
#   data/raw/deam/annotations/.../static_annotations_averaged_songs_1_2000.csv
#   data/raw/deam/audio/<song_id>.mp3
# --------------------------------------------------------------------------- #
def read_deam(root: Path) -> list[dict]:
    base = Path(root) / "deam"
    ann_files = list(base.rglob("static_annotations_averaged_songs*.csv"))
    if not ann_files:
        raise FileNotFoundError(f"expected DEAM static annotations under {base}")
    audio_root = next((p for p in (base / "audio", base / "MEMD_audio") if p.is_dir()), base)

    records = []
    for ann in sorted(ann_files):
        with open(ann, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, skipinitialspace=True):
                sid = (row.get("song_id") or "").strip()
                if not sid:
                    continue
                audio = next((p for p in (audio_root / f"{sid}.mp3", audio_root / f"{sid}.wav")
                              if p.exists()), None)
                if audio is None:
                    continue

                def _num(*keys):
                    for k in keys:
                        v = row.get(k)
                        if v not in (None, ""):
                            try:
                                return float(v)
                            except ValueError:
                                pass
                    return None

                valence = _num("valence_mean", " valence_mean")
                arousal = _num("arousal_mean", " arousal_mean")
                records.append({
                    "track_id": f"deam_{sid}",
                    "artist": f"deam_{sid}",
                    "audio_path": str(audio),
                    "genre": None,
                    "tags": [],
                    "caption": "a music excerpt annotated for valence and arousal",
                    "lyrics": "",
                    "valence": valence,
                    "arousal": arousal,
                })
    return records


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
READERS = {
    "gtzan": read_gtzan,
    "fma_small": lambda root, **kw: read_fma(root, "small"),
    "fma_medium": lambda root, **kw: read_fma(root, "medium"),
    "magnatagatune": read_magnatagatune,
    "musiccaps": read_musiccaps,
    "deam": read_deam,
}


def stratified_subsample(records: list[dict], n_total: int, seed: int = 42,
                         verbose: bool = True) -> list[dict]:
    """Cut a corpus to `n_total` tracks while preserving its shape.

    Strata are (official split x genre), so the subset keeps both the corpus's
    train/val/test proportions and its genre balance -- a naive head-of-list cut
    silently skews both, and on FMA would hand you whichever genres sort first.

    This runs on the *record list*, before any audio is decoded, because
    decoding is the expensive step: subsampling afterwards would waste the hours
    it is meant to save.

    Artists cannot leak as a result of this: strata never cross a split
    boundary, so dropping tracks inside a split cannot move an artist across one.
    """
    if n_total <= 0 or n_total >= len(records):
        return records

    rng = np.random.default_rng(seed)
    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        key = (r.get("split") or "_", r.get("genre") or "_")
        strata.setdefault(key, []).append(i)

    # Largest-remainder allocation, so the quotas sum to exactly n_total.
    total = len(records)
    exact = {k: len(v) * n_total / total for k, v in strata.items()}
    quota = {k: int(v) for k, v in exact.items()}
    remainder = n_total - sum(quota.values())
    for k in sorted(strata, key=lambda k: exact[k] - quota[k], reverse=True)[:remainder]:
        quota[k] += 1

    keep: list[int] = []
    for key, idxs in strata.items():
        take = min(quota[key], len(idxs))
        if take <= 0:
            continue
        keep.extend(rng.choice(idxs, size=take, replace=False).tolist())

    keep.sort()
    subset = [records[i] for i in keep]

    if verbose:
        from collections import Counter
        print(f"  subsampled {len(records)} -> {len(subset)} tracks "
              f"(stratified by split x genre, seed={seed})")
        print(f"    splits: {dict(Counter(r.get('split') or '_' for r in subset))}")
        genres = Counter(r.get("genre") or "_" for r in subset)
        print(f"    genres: {dict(sorted(genres.items()))}")
    return subset


def iter_source(cfg, raw_root: Path):
    """Yield (waveform, record) for the configured source.

    `synthetic` composes audio in memory; every other source decodes the file
    named by the record. Unreadable files are skipped with a warning rather
    than aborting a multi-hour preprocessing run.
    """
    source = cfg["dataset"]["source"]
    sr = cfg["audio"]["sample_rate"]
    duration = cfg["dataset"].get("clip_seconds", 30.0)

    if source == "synthetic":
        from data.synthetic import generate_corpus

        yield from generate_corpus(
            n_tracks=cfg["dataset"]["n_tracks"],
            duration=duration,
            seed=cfg.get("seed", 42),
            sr=sr,
        )
        return

    if source not in READERS:
        raise ValueError(f"unknown dataset.source={source!r}; "
                         f"expected synthetic or one of {sorted(READERS)}")

    kwargs = {"top_tags": cfg["dataset"].get("top_tags", 50)} if source == "magnatagatune" else {}
    records = READERS[source](Path(raw_root), **kwargs)
    if not records:
        raise RuntimeError(f"{source}: no usable records found under {raw_root}")
    print(f"  {source}: {len(records)} records on disk")

    # Subsample before decoding, not after -- decoding is what costs the hours.
    subsample = int(cfg["dataset"].get("subsample", 0) or 0)
    records = stratified_subsample(records, subsample, cfg.get("seed", 42))

    for record in records:
        try:
            y = load_audio(record["audio_path"], sample_rate=sr, duration=duration)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            print(f"  [skip] {record['track_id']}: {type(exc).__name__}: {exc}")
            continue
        if y.size < sr:                   # shorter than a second: unusable
            print(f"  [skip] {record['track_id']}: too short ({y.size} samples)")
            continue
        yield np.asarray(y, dtype=np.float32), record
