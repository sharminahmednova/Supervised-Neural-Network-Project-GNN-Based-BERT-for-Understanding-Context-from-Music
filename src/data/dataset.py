"""Preprocessing pipeline and the torch dataset the trainers consume.

`preprocess(cfg)` runs the spec's section-3 pipeline end to end:

    source -> waveform -> log-mel / chroma / MFCC -> segments -> graph
           -> BERT tokenisation of the paired text
           -> multi-hot tag vector (+ optional valence/arousal targets)
           -> data/processed/graphs/<track_id>.pt

then writes a grouped train/val/test split to data/splits/. Splitting is
*grouped by artist* by default, so no artist appears on both sides of the
train/test boundary -- the leakage the spec calls out for FMA.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from audio_features import extract_track_features
from graph_builder import build_graph, graph_summary
from utils import load_json, save_json


# --------------------------------------------------------------------------- #
# Label space
# --------------------------------------------------------------------------- #
class LabelSpace:
    """Tag vocabulary plus multi-hot encoding, persisted alongside the graphs."""

    def __init__(self, tags: list[str], genres: list[str] | None = None):
        self.tags = list(tags)
        self.tag_to_idx = {t: i for i, t in enumerate(self.tags)}
        self.genres = list(genres or [])
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

    def __len__(self) -> int:
        return len(self.tags)

    def encode(self, tags) -> np.ndarray:
        vec = np.zeros(len(self.tags), dtype=np.float32)
        for t in tags:
            i = self.tag_to_idx.get(t)
            if i is not None:
                vec[i] = 1.0
        return vec

    def decode(self, vec, threshold: float = 0.5) -> list[str]:
        vec = np.asarray(vec).reshape(-1)
        return [self.tags[i] for i in np.where(vec >= threshold)[0]]

    def topk(self, scores, k: int = 5) -> list[tuple[str, float]]:
        scores = np.asarray(scores).reshape(-1)
        order = np.argsort(-scores)[:k]
        return [(self.tags[i], float(scores[i])) for i in order]

    def to_dict(self) -> dict:
        return {"tags": self.tags, "genres": self.genres}

    @classmethod
    def from_dict(cls, blob: dict) -> "LabelSpace":
        return cls(blob["tags"], blob.get("genres"))


# --------------------------------------------------------------------------- #
# Text assembly
# --------------------------------------------------------------------------- #
def partition_tags(tags: list[str], text_fraction: float, seed: int = 42) -> tuple:
    """Split a tag vocabulary into a text-side half and a label-side half.

    Corpora like GTZAN and MagnaTagATune have no free-text annotation, so the
    only available "text" is the tag set -- which is also the prediction target.
    Feeding it in whole makes Task 1 trivially perfect (measured: val Macro-F1
    = 1.0000 after one epoch on GTZAN, because the caption literally reads
    "A blues track tagged blues"), and the Task 3 ablation then compares
    nothing.

    Partitioning the vocabulary fixes that: the model is told some of a clip's
    tags and must infer the rest, from text plus audio structure. The two sets
    are disjoint, so no label can leak through the text branch. This is the
    spec's "caption -> tag proxy task" applied to a corpus with no captions.

    Tags are partitioned by SYNONYM CLUSTER, not individually. MagnaTagATune's
    vocabulary contains overlapping tag strings -- "male", "male vocal" and
    "male voice" are all separate tags -- so a per-tag random split puts
    "male vocal" in the text and "male" in the target, and the text then
    literally contains the answer. Measured on a 2,500-clip subsample: 156 of
    2,499 records (6%) leaked a target tag as a word of their own text.

    Clustering is deliberately LEXICAL (shared tokens after singularisation,
    plus a small list of exact synonyms that happen not to share a token).
    It is not semantic: "techno" and "electronic" stay in different clusters
    because entailment-based clustering collapses most of the vocabulary into
    one blob and leaves nothing to partition. The residual correlation between
    semantically related tags on opposite sides is a real limitation and is
    reported as one.

    Returns (text_side, label_side).
    """
    if text_fraction <= 0.0:
        return [], list(tags)

    clusters = cluster_tags(tags)
    rng = np.random.default_rng(seed)
    order = list(range(len(clusters)))
    rng.shuffle(order)

    target = len(tags) * text_fraction
    text_side: list[str] = []
    for ci in order:
        if len(text_side) >= target:
            break
        text_side.extend(clusters[ci])

    text_set = set(text_side)
    label_side = [t for t in tags if t not in text_set]

    # Both sides must be non-empty; with very few clusters, fall back to
    # donating the smallest cluster rather than producing an empty side.
    if not label_side or not text_side:
        if len(clusters) < 2:
            return [], list(tags)          # one cluster: partitioning is impossible
        smallest = min(clusters, key=len)
        if not label_side:
            text_side = [t for t in text_side if t not in set(smallest)]
            label_side = sorted(smallest)
        else:
            text_side = sorted(smallest)
            label_side = [t for t in tags if t not in set(smallest)]

    return sorted(text_side), sorted(label_side)


# Exact synonyms that share no token, so token overlap alone will not group
# them. Corpus-specific to MagnaTagATune's vocabulary; harmless elsewhere.
_SYNONYM_GROUPS = [
    {"female", "woman", "women", "girl", "lady"},
    {"male", "man", "men", "guy"},
    {"vocal", "voice", "singing", "sing", "singer", "vocalist", "vocals"},
    {"instrumental", "no singing"},
    {"classic", "classical"},
    {"choir", "choral", "chorus"},
    {"opera", "operatic"},
    {"synth", "synthesizer", "synthesiser"},
    {"electronic", "electronica", "electro"},
    {"quiet", "soft", "silence"},
    {"loud", "noisy", "noise"},
    {"weird", "strange"},
    {"harpsicord", "harpsichord"},
    {"solo", "single"},
]

# Tokens too generic to imply a relationship between two tags.
_STOPWORD_TOKENS = {"no", "not", "a", "an", "the", "and", "with", "of", "very",
                    "like", "sounds", "music", "sound"}


def _normalise_token(token: str) -> str:
    token = token.strip().lower()
    # crude singularisation: enough to link beat/beats, string/strings
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def cluster_tags(tags: list[str]) -> list[list[str]]:
    """Group tags that would leak one another if split across text and target.

    Two tags join the same cluster when they share a non-stopword token (after
    singularisation) or appear together in `_SYNONYM_GROUPS`. Union-find, so
    relationships chain: "male vocal" links to "vocal", which links to
    "female vocal", which links to "female".
    """
    tags = list(tags)
    parent = list(range(len(tags)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    token_sets = []
    for tag in tags:
        tokens = {_normalise_token(t) for t in re.split(r"[\s\-_/]+", tag)}
        token_sets.append({t for t in tokens if t and t not in _STOPWORD_TOKENS})

    synonym_of = {}
    for gi, group in enumerate(_SYNONYM_GROUPS):
        for word in group:
            synonym_of[_normalise_token(word)] = gi

    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            if token_sets[i] & token_sets[j]:
                union(i, j)
                continue
            groups_i = {synonym_of[t] for t in token_sets[i] if t in synonym_of}
            groups_j = {synonym_of[t] for t in token_sets[j] if t in synonym_of}
            if groups_i & groups_j:
                union(i, j)

    buckets: dict[int, list[str]] = {}
    for i, tag in enumerate(tags):
        buckets.setdefault(find(i), []).append(tag)
    return [sorted(v) for v in buckets.values()]


def text_leaks_labels(texts: list[str], label_side: list[str]) -> list[tuple]:
    """Records whose text contains a target tag as a whole word.

    Run after partitioning: a non-empty result means the text branch can read
    off part of its own target and every Task 1/3 number is inflated.
    """
    patterns = [(tag, re.compile(rf"\b{re.escape(tag)}\b", re.IGNORECASE))
                for tag in label_side]
    hits = []
    for i, text in enumerate(texts):
        for tag, pattern in patterns:
            if pattern.search(text):
                hits.append((i, tag, text))
                break
    return hits


def compose_text(record: dict, field: str = "caption",
                 text_side: list | None = None) -> str:
    """Build the BERT input string from a record, per `dataset.text_field`.

    When `text_side` is given, only those tags may appear in the text; the rest
    are prediction targets and must not be mentioned.
    """
    caption = (record.get("caption") or "").strip()
    lyrics = (record.get("lyrics") or "").strip()

    record_tags = [t.strip().lower() for t in (record.get("tags") or []) if t]
    if text_side is not None:
        allowed = set(text_side)
        visible = [t for t in record_tags if t in allowed]
        if not visible:
            return "a music track with no listed descriptors"
        return f"A music track described as {', '.join(sorted(visible))}."

    tags = ", ".join(record.get("tags") or [])

    if field == "caption":
        text = caption
    elif field == "lyrics":
        text = lyrics or caption
    elif field == "tags":
        text = tags
    elif field == "caption+tags":
        text = f"{caption} Tags: {tags}." if caption else tags
    elif field == "caption+lyrics":
        text = f"{caption} {lyrics}".strip()
    else:
        raise ValueError(f"unknown dataset.text_field={field!r}")
    return text or "a music track"


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def official_split(records: list[dict]) -> dict | None:
    """Use the corpus's own partition when every record carries one.

    Spec section 3, step 5 asks for the official FMA / MagnaTagATune splits.
    Honouring them is what makes a reported Macro-F1 comparable to published
    numbers; a random split of the same corpus is not the same benchmark.
    Returns None if any record lacks a split, so the caller can fall back.
    """
    buckets: dict = {"train": [], "val": [], "test": []}
    for r in records:
        name = r.get("split")
        if name not in buckets:
            return None
        buckets[name].append(r["track_id"])
    if not all(buckets[s] for s in ("train", "val", "test")):
        return None
    return buckets


def make_split(records: list[dict], cfg_split: dict, seed: int = 42) -> tuple:
    """Official partition if available and allowed, else artist-grouped.

    Returns (split_dict, strategy_name).
    """
    if cfg_split.get("use_official", True):
        official = official_split(records)
        if official is not None:
            return official, "official"
    grouped = grouped_split(records, cfg_split, cfg_split.get("group_by"), seed)
    return grouped, f"grouped_by_{cfg_split.get('group_by') or 'none'}"


def grouped_split(records: list[dict], ratios: dict, group_by: str | None = "artist",
                  seed: int = 42) -> dict:
    """Split track ids into train/val/test, keeping every group intact.

    With `group_by=None` this degrades to a plain shuffled split; with
    "artist" no artist's tracks straddle a split boundary.
    """
    rng = np.random.default_rng(seed)
    if group_by:
        groups: dict[str, list[str]] = {}
        for r in records:
            groups.setdefault(str(r.get(group_by) or r["track_id"]), []).append(r["track_id"])
    else:
        groups = {r["track_id"]: [r["track_id"]] for r in records}

    keys = sorted(groups)
    rng.shuffle(keys)
    total = sum(len(groups[k]) for k in keys)
    targets = {
        "train": ratios.get("train", 0.7) * total,
        "val": ratios.get("val", 0.15) * total,
        "test": ratios.get("test", 0.15) * total,
    }

    # Stratify by label so every genre reaches every split. Without this, a
    # corpus whose groups are genre-pure (GTZAN: one group = one genre's index
    # decade) can send all of a genre's groups to train, leaving that genre
    # absent from test -- the model is then scored on labels it could never
    # produce there. Groups still move whole; only their *order* changes.
    track_label = {r["track_id"]: (r.get("genre") or "") for r in records}
    group_label = {k: track_label.get(v[0], "") for k, v in groups.items()}
    by_label: dict[str, list[str]] = {}
    for key in keys:
        by_label.setdefault(group_label[key], []).append(key)

    # Interleave labels so consecutive assignments rotate through genres.
    ordered: list[str] = []
    pools = [by_label[lab] for lab in sorted(by_label)]
    while any(pools):
        for pool in pools:
            if pool:
                ordered.append(pool.pop())
    keys = ordered

    # Assign whole groups to whichever bucket is furthest below its quota. Groups
    # must stay intact -- moving individual tracks is exactly how artist leakage
    # gets reintroduced, so every rebalancing step below is group-granular too.
    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    sizes = {k: 0 for k in assigned}
    for key in keys:
        bucket = max(assigned, key=lambda b: targets[b] - sizes[b])
        assigned[bucket].append(key)
        sizes[bucket] += len(groups[key])

    # A tiny corpus can still leave val/test empty: donate a whole group from
    # the most over-quota bucket, provided there are groups to spare.
    for bucket in ("val", "test"):
        if assigned[bucket]:
            continue
        donor = max(assigned, key=lambda b: len(assigned[b]))
        if len(assigned[donor]) < 2:
            continue
        smallest = min(assigned[donor], key=lambda k: len(groups[k]))
        assigned[donor].remove(smallest)
        assigned[bucket].append(smallest)

    return {name: [t for key in group_keys for t in groups[key]]
            for name, group_keys in assigned.items()}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess(cfg, raw_root: Path, processed_root: Path, splits_root: Path,
               limit: int | None = None, verbose: bool = True) -> dict:
    """Run the full preprocessing pipeline and persist graphs + splits."""
    from transformers import AutoTokenizer

    from data.adapters import iter_source

    graph_dir = Path(processed_root) / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model_name"])
    max_length = cfg["text"]["max_length"]
    text_field = cfg["dataset"].get("text_field", "caption")

    # -- pass 1: features, graphs, and the raw records (tags not yet a vocab) --
    staged: list[dict] = []
    tag_counter: Counter = Counter()
    genre_set: set[str] = set()

    for n, (waveform, record) in enumerate(iter_source(cfg, Path(raw_root))):
        if limit is not None and n >= limit:
            break
        tf = extract_track_features(waveform, record["track_id"], cfg)
        graph = build_graph(tf, cfg)

        tags = list(record.get("tags") or [])
        if record.get("genre"):
            genre_set.add(record["genre"])
            tags.append(record["genre"])
        tags = sorted(set(t.strip().lower() for t in tags if t and str(t).strip()))
        tag_counter.update(tags)

        staged.append({"graph": graph, "record": record, "tags": tags,
                       "summary": graph_summary(graph)})
        if verbose and (n + 1) % 25 == 0:
            print(f"  featurised {n + 1} tracks...")

    if not staged:
        raise RuntimeError("preprocessing produced no tracks -- check dataset.source "
                           "and the contents of data/raw/")

    # -- label space: keep tags seen often enough to be scoreable --------------
    min_count = int(cfg["dataset"].get("min_tag_count", 1))
    top_k = int(cfg["dataset"].get("top_tags", 0)) or None
    kept = [t for t, c in tag_counter.most_common() if c >= min_count]
    if top_k:
        kept = kept[:top_k]
    text_fraction = float(cfg["dataset"].get("text_tag_fraction", 0.0) or 0.0)
    text_side, label_side = partition_tags(sorted(kept), text_fraction,
                                           cfg.get("seed", 42))
    label_space = LabelSpace(label_side, sorted(genre_set))
    if not len(label_space):
        raise RuntimeError(f"no tag survived min_tag_count={min_count}")
    if text_side:
        print(f"  tag partition: {len(text_side)} tags -> text side, "
              f"{len(label_side)} -> prediction targets (disjoint)")

    for item in staged:
        item["text"] = compose_text(item["record"], text_field,
                                    text_side if text_side else None)

    # The text branch must not be able to read its own target off its input.
    # This is checked, not assumed: a per-tag random partition of MTAT's
    # vocabulary leaked a target tag into 6% of records because "male vocal"
    # and "male" are separate tags.
    if text_side:
        leaks = text_leaks_labels([s["text"] for s in staged], label_side)
        if leaks:
            examples = "\n".join(f'    {tag!r} in {text!r}'
                                 for _, tag, text in leaks[:5])
            raise RuntimeError(
                f"{len(leaks)} of {len(staged)} records have a target tag as a "
                f"word of their own text:\n{examples}\n"
                "The text branch could read part of its own answer, which "
                "inflates every Task 1/3 number. This normally means two tags "
                "belong in the same synonym cluster -- see "
                "dataset.cluster_tags and _SYNONYM_GROUPS.")
        print(f"  leak check: OK (no target tag appears in any record's text)")

        # How often is the text branch actually given anything? MagnaTagATune
        # is sparse -- many clips carry only one or two tags -- so after
        # partitioning a sizeable share of records have no text-side tag at
        # all and are effectively graph-only. That share bounds how much the
        # text branch can contribute, so it is reported rather than buried.
        blank = sum(1 for s in staged if "no listed descriptors" in s["text"])
        print(f"  text coverage: {len(staged) - blank}/{len(staged)} records "
              f"({100 * (1 - blank / max(len(staged), 1)):.1f}%) have at least "
              f"one text-side tag")

        # Textless records share one identical placeholder string, so their
        # text embeddings are identical and every pair among them ties exactly.
        # For tagging that is merely uninformative, but for Task 4 retrieval it
        # is corrupting: 35% identical texts produced ~61 ties per query, and a
        # collapsed encoder still scored R@1 = 0.136 under optimistic ranking.
        # Dropping them makes retrieval well-posed.
        if blank and cfg["dataset"].get("drop_textless", False):
            staged = [s for s in staged if "no listed descriptors" not in s["text"]]
            print(f"  dropped {blank} textless records "
                  f"(dataset.drop_textless) -> {len(staged)} remain")
            if not staged:
                raise RuntimeError(
                    "every record was textless -- lower dataset.text_tag_fraction "
                    "or set dataset.drop_textless=false")
        elif blank > 0.25 * len(staged):
            print(f"  NOTE: {100 * blank / len(staged):.0f}% of records have no "
                  f"text-side tag. They all share one placeholder string, so "
                  f"their\n        text embeddings are identical and tie in "
                  f"retrieval. For Task 4, set dataset.drop_textless=true.")

    # -- pass 2: tokenise, attach targets, write one .pt per track ------------
    index: list[dict] = []
    texts = [s["text"] for s in staged]
    encoded = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]

    for i, item in enumerate(staged):
        rec, graph = item["record"], item["graph"]
        y = torch.from_numpy(label_space.encode(item["tags"])).unsqueeze(0)

        valence, arousal = rec.get("valence"), rec.get("arousal")
        has_va = valence is not None and arousal is not None
        graph.y = y
        graph.va = torch.tensor([[float(valence or 0.0), float(arousal or 0.0)]],
                                dtype=torch.float32)
        graph.va_mask = torch.tensor([1.0 if has_va else 0.0], dtype=torch.float32)
        # .clone() is load-bearing, not defensive. The tokenizer encodes the
        # whole corpus in one call, so input_ids[i] is a *view* into that
        # (n_tracks, max_length) tensor -- and torch.save serialises a view's
        # entire underlying storage. Without the clone every per-track .pt file
        # embeds the tokenised text of the full corpus: measured 2.0 MB per
        # 6-node graph instead of ~10 KB, which at 8,000 MagnaTagATune clips
        # would be ~131 GB of graphs rather than ~0.2 GB.
        graph.input_ids = input_ids[i].unsqueeze(0).clone()
        graph.attention_mask = attention_mask[i].unsqueeze(0).clone()
        graph.track_id = rec["track_id"]
        graph.artist = str(rec.get("artist") or rec["track_id"])
        graph.genre = rec.get("genre") or ""
        graph.text = item["text"]

        torch.save(graph, graph_dir / f"{rec['track_id']}.pt")
        index.append({
            "track_id": rec["track_id"],
            "artist": graph.artist,
            "genre": graph.genre,
            "mood": rec.get("mood", ""),
            "tags": item["tags"],
            "text": item["text"],
            "valence": valence,
            "arousal": arousal,
            "split": rec.get("split"),
            # Kept so downstream tools can get back to the source audio without
            # re-deriving the corpus layout -- the Task 4 listening study has to
            # export real clips for people to hear. None for synthetic tracks,
            # which are generated rather than read from disk.
            "audio_path": rec.get("audio_path"),
            **{k: item["summary"][k] for k in ("num_nodes", "num_edges", "avg_degree")},
        })

    split, strategy = make_split(index, cfg["split"], cfg.get("seed", 42))

    meta = {
        "source": cfg["dataset"]["source"],
        "n_tracks": len(index),
        "graph_kind": cfg["graph"]["kind"],
        "node_feature_dim": int(staged[0]["graph"].x.size(1)),
        "text_field": text_field,
        "text_model": cfg["text"]["model_name"],
        "max_length": max_length,
        "num_tags": len(label_space),
        "text_tag_fraction": text_fraction,
        "text_coverage": round(
            1.0 - sum(1 for s in staged if "no listed descriptors" in s["text"])
            / max(len(staged), 1), 4),
        "text_side_tags": text_side,
        "label_side_tags": label_side,
        "has_emotion": bool(sum(1 for r in index if r["valence"] is not None)),
        "tag_counts": dict(tag_counter.most_common()),
        "avg_nodes": round(float(np.mean([r["num_nodes"] for r in index])), 2),
        "avg_edges": round(float(np.mean([r["num_edges"] for r in index])), 2),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "split_strategy": strategy,
    }

    save_json(label_space.to_dict(), Path(processed_root) / "label_space.json")
    save_json(index, Path(processed_root) / "index.json")
    save_json(meta, Path(processed_root) / "meta.json")
    Path(splits_root).mkdir(parents=True, exist_ok=True)
    for name, ids in split.items():
        save_json(ids, Path(splits_root) / f"{name}.json")

    if verbose:
        print(f"\n  {len(index)} tracks -> {graph_dir}")
        print(f"  tags: {len(label_space)}  |  avg nodes {meta['avg_nodes']} "
              f"edges {meta['avg_edges']}")
        print(f"  split: {meta['split_sizes']}  (strategy: {strategy})")
        if strategy == "official":
            # The corpus owns this partition; report any artist overlap it
            # contains but do not override it -- comparability is the point.
            _report_leakage(index, split, cfg["split"].get("group_by"))
        else:
            _assert_no_leakage(index, split, cfg["split"].get("group_by"), verbose)
    return meta


def _report_leakage(index: list[dict], split: dict, group_by: str | None) -> None:
    """Measure artist overlap in a split we did not create, without changing it.

    The official FMA/MagnaTagATune partitions are not artist-disjoint. The spec
    asks for both the official splits *and* no artist leakage, which cannot
    always hold at once -- so honour the official split and state the overlap,
    rather than silently resplitting and reporting an incomparable number.
    """
    if not group_by:
        return
    by_id = {r["track_id"]: r for r in index}
    groups = {name: {by_id[t][group_by] for t in ids if t in by_id}
              for name, ids in split.items()}
    overlaps = {f"{a}/{b}": len(groups[a] & groups[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    total = sum(overlaps.values())
    if total:
        print(f"  NOTE: official split shares {group_by}s across partitions: {overlaps}")
        print(f"        this is the corpus's own partition -- kept for comparability.")
        print(f"        For an artist-disjoint split instead: --set split.use_official=false")
    else:
        print(f"  leakage check: OK ({group_by} disjoint in the official split)")


def _assert_no_leakage(index: list[dict], split: dict, group_by: str | None,
                       verbose: bool = True) -> None:
    """Verify the grouping key really is disjoint across splits."""
    if not group_by:
        return
    by_id = {r["track_id"]: r for r in index}
    groups = {name: {by_id[t][group_by] for t in ids if t in by_id}
              for name, ids in split.items()}
    overlaps = {
        f"{a}/{b}": sorted(groups[a] & groups[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        if groups[a] & groups[b]
    }
    if overlaps:
        raise RuntimeError(f"{group_by} leakage across splits: {overlaps}")
    if verbose:
        print(f"  leakage check: OK ({group_by} disjoint across train/val/test)")


# --------------------------------------------------------------------------- #
# Dataset / loader
# --------------------------------------------------------------------------- #
class MusicContextDataset(Dataset):
    """Loads the preprocessed graphs for one split."""

    def __init__(self, processed_root, splits_root, split: str = "train",
                 in_memory: bool = True):
        self.graph_dir = Path(processed_root) / "graphs"
        self.label_space = LabelSpace.from_dict(
            load_json(Path(processed_root) / "label_space.json"))
        self.meta = load_json(Path(processed_root) / "meta.json")
        index = {r["track_id"]: r for r in load_json(Path(processed_root) / "index.json")}

        ids = load_json(Path(splits_root) / f"{split}.json")
        self.track_ids = [t for t in ids if (self.graph_dir / f"{t}.pt").exists()]
        self.records = [index[t] for t in self.track_ids if t in index]
        self.split = split
        self._cache = {}
        self.in_memory = in_memory
        if in_memory:
            self._cache = {t: self._load(t) for t in self.track_ids}

    def _load(self, track_id: str) -> Data:
        # weights_only=False: these are our own PyG Data objects, which carry
        # python lists/str attributes that the restricted unpickler rejects.
        return torch.load(self.graph_dir / f"{track_id}.pt", weights_only=False)

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int) -> Data:
        tid = self.track_ids[idx]
        return self._cache[tid] if self.in_memory else self._load(tid)

    # -- convenience views used by the trainers and the report ---------------
    @property
    def num_tags(self) -> int:
        return len(self.label_space)

    @property
    def node_feature_dim(self) -> int:
        return int(self.meta["node_feature_dim"])

    def label_matrix(self) -> np.ndarray:
        return np.stack([self[i].y.numpy().reshape(-1) for i in range(len(self))])

    def pos_weight(self, cap: float = 20.0) -> torch.Tensor:
        """Per-tag positive weight for BCEWithLogitsLoss on a sparse label set."""
        y = self.label_matrix()
        pos = y.sum(axis=0)
        neg = y.shape[0] - pos
        w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
        return torch.tensor(np.clip(w, 0.1, cap), dtype=torch.float32)


def collate_graphs(batch: list[Data]) -> Batch:
    """PyG batching; `follow_batch` is unnecessary since text is one row/graph."""
    return Batch.from_data_list(batch)


def make_loader(dataset: MusicContextDataset, batch_size: int, shuffle: bool,
                num_workers: int = 0):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_graphs,
                      drop_last=False)
