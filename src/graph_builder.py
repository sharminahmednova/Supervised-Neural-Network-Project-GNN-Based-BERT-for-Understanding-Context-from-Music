"""Music structure graphs G = (V, E) for the GNN encoder (spec section 3.3).

Two graph families, plus their union:

segment graph
    nodes = time segments (5-10 s windows or beat-synchronous spans)
    edges = temporal adjacency  +  cosine similarity of MFCC/chroma > tau
            (and a k-NN backstop so no segment is ever isolated)

chord-transition graph
    nodes = unique chords, recognised by matching each segment's chroma vector
            against 24 major/minor triad templates
    edges = observed chord transitions, weighted by occurrence count

Both are emitted as `torch_geometric.data.Data` with `edge_attr` carrying the
edge weight and `edge_type` distinguishing temporal / similarity / transition
edges -- the fusion model's attention analysis reads `edge_type` back out.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from audio_features import TrackFeatures

# Edge type codes, stored on Data.edge_type
EDGE_TEMPORAL = 0
EDGE_SIMILARITY = 1
EDGE_TRANSITION = 2
EDGE_SELF = 3
EDGE_TYPE_NAMES = {
    EDGE_TEMPORAL: "temporal",
    EDGE_SIMILARITY: "similarity",
    EDGE_TRANSITION: "transition",
    EDGE_SELF: "self",
}

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------- #
# Chord recognition
# --------------------------------------------------------------------------- #
def chord_templates() -> tuple[np.ndarray, list[str]]:
    """24 binary triad templates (12 major + 12 minor), L2-normalised."""
    templates, names = [], []
    for root in range(12):
        for quality, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            vec = np.zeros(12, dtype=np.float32)
            for step in intervals:
                vec[(root + step) % 12] = 1.0
            templates.append(vec / np.linalg.norm(vec))
            names.append(f"{PITCH_CLASSES[root]}:{quality}")
    return np.stack(templates), names


_TEMPLATES, _TEMPLATE_NAMES = chord_templates()


def recognise_chords(chroma_segments: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map each segment's chroma vector to its best-matching triad template.

    Returns (chord_index_per_segment, chord_names).
    """
    x = np.asarray(chroma_segments, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-9)
    scores = x @ _TEMPLATES.T            # (n_segments, 24) cosine similarity
    idx = scores.argmax(axis=1)
    return idx.astype(np.int64), _TEMPLATE_NAMES


# --------------------------------------------------------------------------- #
# Edge construction helpers
# --------------------------------------------------------------------------- #
def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.maximum(norms, 1e-9)
    return np.clip(xn @ xn.T, -1.0, 1.0)


def effective_knn(requested: int, n_nodes: int) -> int:
    """Scale k down on small graphs.

    A 30 s clip cut into 5 s windows is a *six node* graph, and a fixed k=4
    backstop there wires almost every pair: the graph comes out ~95% dense, at
    which point message passing is just mean pooling and the structural premise
    of Tasks 2 and 3 is gone.

    The backstop exists only to guarantee connectivity, which k=1 already does,
    so it stays at 1 until the graph is big enough (beat-synchronous segments
    give 30-70 nodes) for a wider k to mean something. Past that point the
    similarity threshold -- not this -- is what decides the edge set.
    """
    if requested <= 0 or n_nodes < 2:
        return 0
    return max(1, min(requested, n_nodes // 8))


def _similarity_edges(sim: np.ndarray, threshold: float, knn: int) -> list:
    """Upper-triangular (i<j) similarity edges above tau, plus a k-NN backstop."""
    n = sim.shape[0]
    pairs: dict[tuple[int, int], float] = {}
    if n < 2:
        return []

    rows, cols = np.where(np.triu(sim, k=1) > threshold)
    for i, j in zip(rows.tolist(), cols.tolist()):
        pairs[(i, j)] = float(sim[i, j])

    # k-NN backstop: guarantees connectivity when tau is set aggressively high.
    k = effective_knn(knn, n)
    if k > 0:
        masked = sim.copy()
        np.fill_diagonal(masked, -np.inf)
        for i in range(n):
            for j in np.argpartition(-masked[i], k - 1)[:k].tolist():
                a, b = (i, j) if i < j else (j, i)
                if a != b:
                    pairs.setdefault((a, b), float(sim[a, b]))

    return [(i, j, w) for (i, j), w in pairs.items()]


def _to_tensors(n_nodes: int, edges: list, undirected: bool, self_loops: bool):
    """(src, dst, weight, type) tuples -> edge_index / edge_attr / edge_type."""
    src, dst, weight, etype = [], [], [], []
    for i, j, w, t in edges:
        src.append(i)
        dst.append(j)
        weight.append(w)
        etype.append(t)
        if undirected:
            src.append(j)
            dst.append(i)
            weight.append(w)
            etype.append(t)
    if self_loops:
        for i in range(n_nodes):
            src.append(i)
            dst.append(i)
            weight.append(1.0)
            etype.append(EDGE_SELF)
    if not src:  # isolated single-node graph
        src, dst, weight, etype = [0], [0], [1.0], [EDGE_SELF]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weight, dtype=torch.float32).unsqueeze(1)
    edge_type = torch.tensor(etype, dtype=torch.long)
    return edge_index, edge_attr, edge_type


# --------------------------------------------------------------------------- #
# Graph builders
# --------------------------------------------------------------------------- #
def build_segment_graph(tf: TrackFeatures, cfg) -> Data:
    """Nodes = segments; edges = temporal adjacency + chroma/MFCC similarity."""
    g = cfg["graph"]
    n = tf.n_segments
    edges: list = []

    if g.get("temporal_edges", True):
        edges += [(i, i + 1, 1.0, EDGE_TEMPORAL) for i in range(n - 1)]

    # Similarity on the concatenated timbre (MFCC) + harmony (chroma) view.
    view = np.concatenate([tf.mfcc_segments, tf.chroma_segments], axis=1)
    sim = _cosine_matrix(view)
    adjacent = {(i, i + 1) for i in range(n - 1)}
    for i, j, w in _similarity_edges(sim, g["similarity_threshold"], g.get("knn", 0)):
        if (i, j) not in adjacent:          # don't duplicate temporal edges
            edges.append((i, j, w, EDGE_SIMILARITY))

    edge_index, edge_attr, edge_type = _to_tensors(
        n, edges, g.get("undirected", True), g.get("self_loops", True))

    data = Data(
        x=torch.from_numpy(tf.node_features),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.edge_type = edge_type
    data.num_nodes = n
    data.track_id = tf.track_id
    data.segment_times = tf.segment_times()
    data.node_kind = "segment"
    return data


def build_chord_graph(tf: TrackFeatures, cfg) -> Data:
    """Nodes = unique chords; edges = transitions weighted by observed count."""
    g = cfg["graph"]
    chord_idx, chord_names = recognise_chords(tf.chroma_segments)
    uniques = sorted(set(chord_idx.tolist()))
    remap = {c: k for k, c in enumerate(uniques)}
    n = len(uniques)

    # Node feature = mean of the segment features assigned to that chord,
    # so chord nodes stay comparable with segment nodes in the hybrid graph.
    feat_dim = tf.feature_dim
    x = np.zeros((n, feat_dim), dtype=np.float32)
    for chord, k in remap.items():
        mask = chord_idx == chord
        x[k] = tf.node_features[mask].mean(axis=0)

    counts: dict[tuple[int, int], float] = {}
    for a, b in zip(chord_idx[:-1].tolist(), chord_idx[1:].tolist()):
        if a == b:
            continue                       # skip self-transitions; self-loops added later
        key = (remap[a], remap[b])
        counts[key] = counts.get(key, 0.0) + 1.0
    total = max(sum(counts.values()), 1.0)
    edges = [(i, j, w / total, EDGE_TRANSITION) for (i, j), w in counts.items()]

    # Chord transitions are directional -- keep them so, regardless of cfg.
    edge_index, edge_attr, edge_type = _to_tensors(
        n, edges, undirected=False, self_loops=g.get("self_loops", True))

    data = Data(x=torch.from_numpy(x), edge_index=edge_index, edge_attr=edge_attr)
    data.edge_type = edge_type
    data.num_nodes = n
    data.track_id = tf.track_id
    data.chord_names = [chord_names[c] for c in uniques]
    data.chord_sequence = [chord_names[c] for c in chord_idx.tolist()]
    data.node_kind = "chord"
    return data


def build_hybrid_graph(tf: TrackFeatures, cfg) -> Data:
    """Segment graph plus chord nodes, linked segment -> its recognised chord."""
    seg = build_segment_graph(tf, cfg)
    chord = build_chord_graph(tf, cfg)
    n_seg = seg.num_nodes

    chord_idx, chord_names = recognise_chords(tf.chroma_segments)
    uniques = sorted(set(chord_idx.tolist()))
    remap = {c: k + n_seg for k, c in enumerate(uniques)}

    x = torch.cat([seg.x, chord.x], dim=0)
    chord_edges = chord.edge_index + n_seg
    membership = torch.tensor(
        [[i for i in range(n_seg)] + [remap[c] for c in chord_idx.tolist()],
         [remap[c] for c in chord_idx.tolist()] + [i for i in range(n_seg)]],
        dtype=torch.long,
    )

    edge_index = torch.cat([seg.edge_index, chord_edges, membership], dim=1)
    edge_attr = torch.cat([
        seg.edge_attr,
        chord.edge_attr,
        torch.ones(membership.size(1), 1, dtype=torch.float32),
    ], dim=0)
    edge_type = torch.cat([
        seg.edge_type,
        chord.edge_type,
        torch.full((membership.size(1),), EDGE_TRANSITION, dtype=torch.long),
    ])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_type = edge_type
    data.num_nodes = x.size(0)
    data.track_id = tf.track_id
    data.chord_names = [chord_names[c] for c in uniques]
    data.node_kind = "hybrid"
    data.n_segment_nodes = n_seg
    return data


def build_graph(tf: TrackFeatures, cfg) -> Data:
    """Dispatch on cfg.graph.kind."""
    kind = cfg["graph"].get("kind", "segment")
    builders = {
        "segment": build_segment_graph,
        "chord": build_chord_graph,
        "hybrid": build_hybrid_graph,
    }
    if kind not in builders:
        raise ValueError(f"unknown graph.kind={kind!r}; expected one of {sorted(builders)}")
    return builders[kind](tf, cfg)


def graph_summary(data: Data) -> dict:
    """Human-readable stats, used by the EDA notebook and the report tables."""
    et = data.edge_type.tolist() if hasattr(data, "edge_type") else []
    by_type = {name: et.count(code) for code, name in EDGE_TYPE_NAMES.items()}
    n, e = int(data.num_nodes), int(data.edge_index.size(1))
    return {
        "track_id": getattr(data, "track_id", None),
        "node_kind": getattr(data, "node_kind", "?"),
        "num_nodes": n,
        "num_edges": e,
        "avg_degree": round(e / max(n, 1), 3),
        "density": round(e / max(n * (n - 1), 1), 4),
        "edges_by_type": by_type,
        "feature_dim": int(data.x.size(1)),
    }
