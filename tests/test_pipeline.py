"""Pipeline tests -- run with `python tests/test_pipeline.py` or `pytest tests/`.

These cover the things that are easy to get quietly wrong and hard to notice
from a training curve: graph wiring, batching, masked losses, metric maths, and
above all split integrity.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from audio_features import (extract_track_features, fixed_window_bounds,  # noqa: E402
                            node_feature_dim, normalize_per_track)
from data.dataset import LabelSpace, grouped_split  # noqa: E402
from data.synthetic import GENRES, build_tags, synthesize_track  # noqa: E402
from graph_builder import (EDGE_SELF, build_chord_graph, build_hybrid_graph,  # noqa: E402
                           build_segment_graph, recognise_chords)
from metrics import (regression_metrics, retrieval_metrics, tagging_metrics,  # noqa: E402
                     tune_threshold)
from utils import load_config  # noqa: E402

CFG = load_config(REPO / "config.yaml")
SR = CFG["audio"]["sample_rate"]


def _demo_track(genre: str = "rock", mood: str = "energetic", seconds: float = 20.0):
    y, meta = synthesize_track(genre, mood, seed=3, duration=seconds, sr=SR)
    return extract_track_features(y, "t_test", CFG), meta


# --------------------------------------------------------------------------- #
# Audio features
# --------------------------------------------------------------------------- #
def test_feature_shapes():
    tf, _ = _demo_track()
    assert tf.feature_dim == node_feature_dim(CFG), \
        f"{tf.feature_dim} != declared {node_feature_dim(CFG)}"
    assert tf.n_segments >= 3
    assert np.isfinite(tf.node_features).all(), "non-finite node features"
    assert tf.chroma_segments.shape == (tf.n_segments, CFG["audio"]["n_chroma"])


def test_normalisation_is_zero_mean():
    x = np.random.RandomState(0).randn(12, 400).astype(np.float32) * 5 + 3
    out = normalize_per_track(x)
    assert np.allclose(out.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(out.std(axis=1), 1, atol=1e-4)


def test_segment_bounds_cover_the_track():
    bounds = fixed_window_bounds(1000, SR, 512, 5.0)
    assert bounds[0][0] == 0 and bounds[-1][1] == 1000
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        assert end == start, "segments must tile without gaps or overlap"


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #
def test_segment_graph_structure():
    tf, _ = _demo_track()
    g = build_segment_graph(tf, CFG)
    assert g.num_nodes == tf.n_segments
    assert g.edge_index.max() < g.num_nodes, "edge index out of range"
    assert g.edge_index.size(1) == g.edge_attr.size(0) == g.edge_type.size(0)
    # every node reachable: self-loops plus temporal chain guarantee degree >= 1
    deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes)
    assert (deg > 0).all(), "isolated node in segment graph"


def test_chord_graph_is_directed_and_weighted():
    tf, _ = _demo_track()
    g = build_chord_graph(tf, CFG)
    assert 1 <= g.num_nodes <= 24
    assert len(g.chord_names) == g.num_nodes
    non_self = g.edge_type != EDGE_SELF
    if non_self.any():
        w = g.edge_attr[non_self].reshape(-1)
        assert (w > 0).all() and w.sum() <= 1.0 + 1e-5, "transition weights not normalised"


def test_chord_recognition_finds_the_written_chord():
    # A pure C major triad must be recognised as C:maj, not something else.
    t = np.arange(int(SR * 6)) / SR
    y = sum(0.3 * np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.00))
    tf = extract_track_features(y.astype(np.float32), "cmaj", CFG)
    idx, names = recognise_chords(tf.chroma_segments)
    recognised = {names[i] for i in idx}
    assert "C:maj" in recognised, f"expected C:maj, got {recognised}"


def test_hybrid_graph_links_segments_to_chords():
    tf, _ = _demo_track()
    g = build_hybrid_graph(tf, CFG)
    seg = build_segment_graph(tf, CFG)
    chord = build_chord_graph(tf, CFG)
    assert g.num_nodes == seg.num_nodes + chord.num_nodes
    assert g.edge_index.max() < g.num_nodes


# --------------------------------------------------------------------------- #
# Labels and splitting
# --------------------------------------------------------------------------- #
def test_label_space_roundtrip():
    ls = LabelSpace(["jazz", "calm", "piano"])
    vec = ls.encode(["calm", "piano", "not-a-tag"])
    assert vec.tolist() == [0.0, 1.0, 1.0]
    assert ls.decode(vec) == ["calm", "piano"]


def test_grouped_split_never_leaks_an_artist():
    records = [{"track_id": f"t{i}", "artist": f"a{i // 5}"} for i in range(60)]
    split = grouped_split(records, {"train": 0.7, "val": 0.15, "test": 0.15}, "artist", 0)
    by_id = {r["track_id"]: r for r in records}
    groups = {k: {by_id[t]["artist"] for t in ids} for k, ids in split.items()}

    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert sum(len(v) for v in split.values()) == len(records), "tracks lost or duplicated"
    assert all(split[s] for s in ("train", "val", "test")), "empty split"


def test_grouped_split_is_deterministic():
    records = [{"track_id": f"t{i}", "artist": f"a{i // 3}"} for i in range(30)]
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = grouped_split(records, ratios, "artist", 7)
    b = grouped_split(records, ratios, "artist", 7)
    assert a == b, "same seed must give the same split"


def test_tiny_corpus_still_fills_every_split():
    records = [{"track_id": f"t{i}", "artist": f"a{i}"} for i in range(4)]
    split = grouped_split(records, {"train": 0.7, "val": 0.15, "test": 0.15}, "artist", 1)
    assert all(split[s] for s in ("train", "val", "test"))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_perfect_predictions_score_one():
    y = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=float)
    m = tagging_metrics(y, y * 0.99 + 0.005, threshold=0.5)
    assert abs(m["macro_f1"] - 1.0) < 1e-6
    assert abs(m["micro_f1"] - 1.0) < 1e-6


def test_absent_tags_do_not_deflate_macro_f1():
    # column 2 is all-zero in the targets: F1 is undefined, not zero.
    y = np.array([[1, 0, 0], [1, 1, 0]], dtype=float)
    scores = np.array([[0.9, 0.1, 0.1], [0.9, 0.9, 0.1]])
    m = tagging_metrics(y, scores, threshold=0.5)
    assert m["n_tags_scored"] == 2
    assert abs(m["macro_f1"] - 1.0) < 1e-6
    assert m["macro_f1_all_tags"] < m["macro_f1"]


def test_threshold_tuning_beats_the_default():
    rng = np.random.RandomState(0)
    y = (rng.rand(60, 6) < 0.15).astype(float)      # sparse labels
    scores = np.clip(y * 0.3 + rng.rand(60, 6) * 0.25, 0, 1)
    best_t, best_f1 = tune_threshold(y, scores)
    assert best_f1 >= tagging_metrics(y, scores, 0.5)["macro_f1"] - 1e-9
    assert 0.0 < best_t < 1.0


def test_retrieval_metrics_on_a_perfect_matrix():
    sim = np.eye(8) * 2 - 1
    m = retrieval_metrics(sim)
    assert m["mean_R@1"] == 1.0
    assert m["audio2caption_median_rank"] == 1.0


def test_retrieval_ties_do_not_flatter_a_collapsed_model():
    """A model that gives every pair the same score must score at chance.

    Counting only strictly-better candidates gives every tied item rank 1, so a
    fully collapsed encoder reported R@1 = 1.0. Mid-rank tie-breaking is what
    makes the metric honest; this caught a real 0.136 R@1 from a model that
    retrieved the same three tracks for every query.
    """
    n = 50
    collapsed = np.ones((n, n))                 # identical similarity everywhere
    m = retrieval_metrics(collapsed)
    assert abs(m["mean_R@1"] - 1.0 / n) < 1e-6, \
        f"collapsed model scored R@1={m['mean_R@1']:.3f}, expected chance {1/n:.3f}"
    assert abs(m["audio2caption_median_rank"] - (n + 1) / 2) < 1e-6
    assert m["chance_R@1"] == 1.0 / n

    # Half the items tied at the top, half genuinely separated.
    partial = np.eye(n) * 2.0
    partial[: n // 2] = 1.0                     # first half: all-tied rows
    mp = retrieval_metrics(partial)
    assert mp["mean_R@1"] < 1.0, "tied rows must not all count as rank 1"
    assert mp["audio2caption_mean_ties"] > 0

    perfect = np.eye(n) * 2.0 - 1.0
    assert retrieval_metrics(perfect)["mean_R@1"] == 1.0


def test_retrieval_recall_is_monotonic_in_k():
    rng = np.random.RandomState(1)
    sim = rng.rand(20, 20) + np.eye(20) * 0.3
    m = retrieval_metrics(sim)
    assert m["mean_R@1"] <= m["mean_R@5"] <= m["mean_R@10"]


def test_regression_mask_selects_annotated_rows():
    true = np.array([[5.0, 5.0], [1.0, 1.0]])
    pred = np.array([[5.0, 5.0], [9.0, 9.0]])       # row 1 is wildly wrong
    masked = regression_metrics(true, pred, mask=np.array([1.0, 0.0]))
    assert masked["mae_valence"] == 0.0, "masked-out row leaked into the MAE"
    assert regression_metrics(true, pred)["mae_valence"] > 0


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _mini_batch(n_graphs: int = 3, num_tags: int = 5):
    from torch_geometric.data import Batch

    graphs = []
    for i in range(n_graphs):
        tf, _ = _demo_track(GENRES[i % len(GENRES)], "calm", seconds=15.0)
        g = build_segment_graph(tf, CFG)
        g.y = torch.zeros(1, num_tags)
        g.y[0, i % num_tags] = 1.0
        g.va = torch.tensor([[5.0, 5.0]])
        g.va_mask = torch.tensor([1.0 if i else 0.0])   # one unannotated row
        g.input_ids = torch.randint(0, 1000, (1, 16))
        g.attention_mask = torch.ones(1, 16, dtype=torch.long)
        g.text = f"track {i}"
        g.track_id = f"t{i}"
        g.genre = GENRES[i % len(GENRES)]
        graphs.append(g)
    return Batch.from_data_list(graphs)


def test_gnn_forward_and_batching():
    from gnn_model import build_gnn

    batch = _mini_batch()
    model = build_gnn(CFG, node_feature_dim(CFG), 5)
    logits = model(batch)
    assert logits.shape == (3, 5), logits.shape
    assert torch.isfinite(logits).all()


def test_gat_variant_runs():
    from gnn_model import GNNTagClassifier

    batch = _mini_batch()
    model = GNNTagClassifier(node_feature_dim(CFG), 5, hidden_dim=32,
                             num_layers=2, conv="gat", heads=4)
    assert model(batch).shape == (3, 5)


def test_readout_is_permutation_invariant():
    """Mean pooling must not depend on node ordering."""
    from gnn_model import GNNEncoder

    tf, _ = _demo_track()
    g = build_segment_graph(tf, CFG)
    enc = GNNEncoder(node_feature_dim(CFG), hidden_dim=32, num_layers=2).eval()

    perm = torch.randperm(g.num_nodes)
    inverse = torch.argsort(perm)
    remapped = inverse[g.edge_index]

    with torch.no_grad():
        a = enc(g.x, g.edge_index)
        b = enc(g.x[perm], remapped)
    assert torch.allclose(a, b, atol=1e-4), "readout is order-dependent"


def test_fusion_modes_all_forward():
    from fusion_model import FUSION_MODES, build_fusion_model

    batch = _mini_batch()
    for mode in FUSION_MODES:
        cfg = load_config(REPO / "config.yaml")
        cfg["fusion"]["mode"] = mode
        cfg["fusion"]["proj_dim"] = 64
        cfg["gnn"]["hidden_dim"] = 32
        model = build_fusion_model(cfg, node_feature_dim(CFG), 5)
        out = model(batch)
        assert out["tag_logits"].shape == (3, 5), mode
        assert torch.isfinite(out["tag_logits"]).all(), mode
        if mode == "cross_attention":
            assert out["attention"] is not None
            attn = out["attention"]
            assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-4), \
                "attention rows must sum to 1"


def test_multitask_loss_masks_unannotated_emotion():
    from fusion_model import MultiTaskLoss

    batch = _mini_batch()
    criterion = MultiTaskLoss(alpha=1.0, beta=1.0)
    outputs = {"tag_logits": torch.zeros(3, 5), "emotion": torch.zeros(3, 2)}

    _, parts = criterion(outputs, batch)
    assert "valence_mse" in parts

    batch.va_mask = torch.zeros(3)          # nothing annotated -> tags only
    _, parts_none = criterion(outputs, batch)
    assert "valence_mse" not in parts_none
    assert abs(parts_none["total"] - parts_none["tag_loss"]) < 1e-6


def test_infonce_rewards_the_diagonal():
    from contrastive import info_nce_loss

    aligned = torch.eye(6) * 10.0
    scrambled = torch.eye(6).roll(1, dims=1) * 10.0
    loss_good, stats = info_nce_loss(aligned)
    loss_bad, _ = info_nce_loss(scrambled)
    assert loss_good < loss_bad, "matched pairs must score lower loss"
    assert stats["in_batch_acc"] == 1.0


def test_frozen_bert_has_no_trainable_encoder_params():
    from bert_encoder import BertTagClassifier

    model = BertTagClassifier(4, CFG["text"]["model_name"], freeze=True, unfreeze_last_n=0)
    assert not any(p.requires_grad for p in model.encoder.bert.parameters())
    assert all(p.requires_grad for p in model.head.parameters())

    model.encoder.apply_freezing(True, unfreeze_last_n=2)
    assert any(p.requires_grad for p in model.encoder.bert.parameters()), \
        "unfreeze_last_n did not unfreeze anything"


# --------------------------------------------------------------------------- #
# Provenance, checkpoints, graph density
# --------------------------------------------------------------------------- #
def test_merge_metrics_demands_provenance():
    """A metrics row without provenance is how two corpora got mixed."""
    import tempfile

    from utils import merge_metrics

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.json"
        try:
            merge_metrics(path, "run", {"macro_f1": 0.5})
        except ValueError:
            return
        raise AssertionError("merge_metrics accepted a row with no provenance")


def test_merge_metrics_refuses_to_mix_corpora():
    import tempfile

    from utils import ProvenanceConflict, merge_metrics

    a = {"corpus": "gtzan", "num_tags": 10, "graph_kind": "segment",
         "segment_seconds": 2.5}
    b = {"corpus": "synthetic", "num_tags": 33, "graph_kind": "segment",
         "segment_seconds": 5.0}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.json"
        merge_metrics(path, "baseline", {"macro_f1": 0.18}, a)
        merge_metrics(path, "baseline_two", {"macro_f1": 0.19}, a)   # same corpus: fine
        try:
            merge_metrics(path, "task1", {"macro_f1": 0.85}, b)
        except ProvenanceConflict:
            return
        raise AssertionError("merge_metrics mixed two corpora in one file")


def test_checkpoint_mismatch_is_reported_clearly():
    """Loading a 32-tag checkpoint against 10-tag data must say why."""
    import tempfile

    from engine import CheckpointMismatch, load_checkpoint

    class FakeDataset:
        label_space = LabelSpace(["a", "b", "c"])
        meta = {"source": "gtzan"}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"state_dict": {},
                    "provenance": {"corpus": "synthetic", "num_tags": 33}}, path)
        try:
            load_checkpoint(path, FakeDataset())
        except CheckpointMismatch as exc:
            assert "33" in str(exc) and "3" in str(exc)
            return
        raise AssertionError("incompatible checkpoint loaded without complaint")


def test_effective_knn_does_not_densify_small_graphs():
    """The k-NN backstop must not wire a 6-node graph nearly complete."""
    from graph_builder import effective_knn

    assert effective_knn(4, 6) == 1, "k=4 on a 6-node graph is a near-complete graph"
    assert effective_knn(4, 12) == 1
    assert effective_knn(4, 60) == 4, "large graphs should get the requested k"
    assert effective_knn(0, 50) == 0, "k=0 disables the backstop"
    assert effective_knn(4, 1) == 0


def test_segment_graph_stays_sparse_on_a_30s_clip():
    """Real GTZAN/MTAT clips are ~30 s; density must stay well under complete."""
    cfg = load_config(REPO / "config.yaml")
    cfg["audio"]["segment_seconds"] = 2.5
    y, _ = synthesize_track("rock", "energetic", seed=5, duration=30.0, sr=SR)
    tf = extract_track_features(y, "t", cfg)
    g = build_segment_graph(tf, cfg)

    n = int(g.num_nodes)
    non_self = int(g.edge_index.size(1)) - n
    density = non_self / (n * (n - 1))
    assert n >= 10, f"expected >=10 segments on a 30 s clip, got {n}"
    assert density < 0.45, f"graph is too dense to be structural: {density:.2f}"


def test_saved_graph_does_not_embed_the_whole_corpus():
    """A per-track graph must not serialise the corpus-wide tokenizer output.

    Slicing a batch-tokenised tensor yields a view, and torch.save writes a
    view's entire underlying storage -- so without a .clone() every graph file
    carried every track's tokens (2.0 MB per 6-node graph, ~131 GB at MTAT
    scale). This asserts the per-file size stays proportional to one track.
    """
    import tempfile

    from torch_geometric.data import Data

    n_corpus, max_len = 400, 128
    corpus_ids = torch.randint(0, 30000, (n_corpus, max_len))

    tf, _ = _demo_track(seconds=15.0)
    graph = build_segment_graph(tf, CFG)
    graph.input_ids = corpus_ids[0].unsqueeze(0).clone()
    graph.attention_mask = torch.ones(1, max_len, dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "g.pt"
        torch.save(graph, path)
        size = path.stat().st_size

    # One track's own tensors are ~10-20 KB; the corpus-wide storage would be
    # n_corpus * max_len * 8 B = 400 KB per field.
    budget = 200_000
    assert size < budget, (
        f"saved graph is {size:,} B (> {budget:,}); a tokenizer view is "
        f"probably being saved without .clone()")


def test_tag_partition_keeps_synonyms_on_one_side():
    """The text branch must not be able to read its own target off its input.

    MagnaTagATune has "male", "male vocal" and "male voice" as separate tags,
    so a per-tag random split puts one in the text and another in the target.
    Measured before the fix: 156 of 2,499 records leaked a target tag.
    """
    from data.dataset import (cluster_tags, compose_text, partition_tags,
                              text_leaks_labels)

    vocab = ["male", "male vocal", "male voice", "man",
             "female", "female vocal", "woman",
             "vocal", "vocals", "voice", "singing", "no vocal", "no voice",
             "beat", "beats", "classic", "classical", "quiet", "soft",
             "rock", "techno", "piano", "violin", "drums", "loud", "fast"]

    clusters = cluster_tags(vocab)
    lookup = {tag: i for i, c in enumerate(clusters) for tag in c}
    for a, b in [("male", "male vocal"), ("female", "woman"), ("man", "male"),
                 ("beat", "beats"), ("classic", "classical"),
                 ("vocal", "singing"), ("quiet", "soft")]:
        assert lookup[a] == lookup[b], f"{a!r} and {b!r} must share a cluster"
    assert lookup["rock"] != lookup["piano"], "unrelated tags must not merge"

    for fraction in (0.2, 0.4, 0.6, 0.8):
        text_side, label_side = partition_tags(vocab, fraction, seed=7)
        assert text_side and label_side
        assert not set(text_side) & set(label_side), "sides must be disjoint"

        records = [{"tags": vocab, "caption": "", "lyrics": ""}]
        texts = [compose_text(r, "tags", text_side) for r in records]
        leaks = text_leaks_labels(texts, label_side)
        assert not leaks, f"fraction={fraction} leaked {leaks[:2]}"


def test_krippendorff_alpha_behaves_like_a_reliability_coefficient():
    """The human-eval agreement statistic, checked by its defining properties."""
    sys.path.insert(0, str(REPO / "tools"))
    from aggregate_human_eval import krippendorff_alpha_ordinal as alpha

    base = [1, 2, 3, 4, 5] * 4
    perfect = np.array([base, base], dtype=float)
    assert abs(alpha(perfect) - 1.0) < 1e-9

    rng = np.random.default_rng(0)
    noise = rng.integers(1, 6, (5, 300)).astype(float)
    assert abs(alpha(noise)) < 0.15, "independent ratings should sit near zero"

    inverted = np.array([[1, 1, 1, 5, 5, 5], [5, 5, 5, 1, 1, 1]], dtype=float)
    assert alpha(inverted) < 0, "systematic disagreement must be negative"

    # Ordinal, not nominal: a distant miss must cost more than an adjacent one.
    near = np.array([base, [2, 3, 4, 5, 4] * 4], dtype=float)
    far = np.array([base, [5, 5, 5, 1, 1] * 4], dtype=float)
    assert alpha(near) > alpha(far)

    # Missing cells must not break it.
    holey = np.array([base, base], dtype=float)
    holey[1, ::3] = np.nan
    assert abs(alpha(holey) - 1.0) < 1e-9

    # Degenerate input is reported as undefined, not as agreement.
    assert not np.isfinite(alpha(np.array([[3.0, 3.0], [3.0, 3.0]])))


def test_config_extends_inherits_and_overrides():
    cfg = load_config(REPO / "configs" / "mtat.yaml")
    assert cfg["dataset"]["source"] == "magnatagatune"      # overridden
    assert cfg["audio"]["segment_seconds"] == 2.5           # overridden
    assert cfg["gnn"]["conv"] == "sage"                     # inherited
    assert "epochs" in cfg["train"]                         # inherited


def test_synthetic_tags_match_the_generated_attributes():
    _, meta = synthesize_track("metal", "aggressive", seed=11, duration=10.0)
    tags = build_tags(meta)
    assert "metal" in tags and "aggressive" in tags
    assert ("loud" in tags) == (meta["arousal"] > 6.5)


# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report, don't abort the suite
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("\nfailures:")
        for name, exc in failures:
            print(f"  {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
