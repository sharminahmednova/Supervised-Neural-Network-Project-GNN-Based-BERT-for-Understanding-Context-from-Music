"""Aggregate results: comparison table, F1-vs-epoch curves, t-SNE, PR curves.

    python src/evaluate.py                 # table + every plot it has data for
    python src/evaluate.py --table-only

Reads whatever results/ already contains, so it can run after a single task or
after `train.py all`. Missing artefacts are reported and skipped, never fatal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from metrics import comparison_table  # noqa: E402
from utils import load_config, load_json, resolve, save_json  # noqa: E402

# Order rows the way the spec's Table 3 does: baselines first, then tasks.
ROW_ORDER = [
    ("baseline_b1_random", "B1: Random tags"),
    ("baseline_b1_prior", "B1: Tag prior"),
    ("baseline_b4_pca_mlp", "B4: PCA + MLP"),
    ("baseline_b2_melcnn", "B2: CNN mel-spec"),
    ("task1_bert", "Task 1: BERT-only"),
    ("task2_gnn_sage", "Task 2: GNN-only (SAGE)"),
    ("task2_gnn_gat", "Task 2: GNN-only (GAT)"),
    ("task3_fusion_bert_only", "Task 3 abl: BERT-only"),
    ("task3_fusion_gnn_only", "Task 3 abl: GNN-only"),
    ("task3_fusion_concat", "Task 3 abl: early concat"),
    ("task3_fusion_cross_attention", "Task 3: GNN-BERT x-attn"),
    ("task4_contrastive", "Task 4: Contrastive"),
]


def _prov_key(prov: dict) -> tuple:
    return (prov.get("corpus"), prov.get("num_tags"), prov.get("graph_kind"),
            prov.get("segment_seconds"))


def build_table(metrics: dict) -> tuple[str, dict]:
    """Assemble the comparison table, refusing to mix corpora in one table.

    A row is only comparable with another row if it was produced on the same
    corpus, label space and graph construction. Silently mixing them is how a
    table ends up claiming a result that was never measured.
    """
    rows = {}
    for key, label in ROW_ORDER:
        if key in metrics:
            rows[label] = metrics[key]
    for key, value in metrics.items():           # anything not in ROW_ORDER
        if (key not in dict(ROW_ORDER) and not key.startswith("_")
                and isinstance(value, dict) and "macro_f1" in value):
            rows.setdefault(key, value)

    groups: dict[tuple, list[str]] = {}
    for label, row in rows.items():
        groups.setdefault(_prov_key(row.get("_provenance") or {}), []).append(label)

    if len(groups) > 1:
        detail = "\n".join(
            f"    {k[0]} ({k[1]} tags, graph={k[2]}, seg={k[3]}s): {', '.join(labels)}"
            for k, labels in groups.items())
        raise SystemExit(
            "results/metrics.json mixes results from different corpora, which "
            "cannot go in one table:\n" + detail +
            "\n\nScore each corpus into its own results directory, e.g.\n"
            "  python src/train.py all --set paths.results=results_mtat")

    header = ""
    prov = next(iter(rows.values()), {}).get("_provenance") if rows else None
    if prov:
        header = (f"corpus: {prov.get('corpus')}  |  tracks: {prov.get('n_tracks')}  |  "
                  f"tags: {prov.get('num_tags')}  |  graph: {prov.get('graph_kind')}  |  "
                  f"segment: {prov.get('segment_seconds')}s\n"
                  f"split: {prov.get('split_sizes')} ({prov.get('split_strategy')})\n\n")
    return header + comparison_table(rows), rows


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_training_curves(results_dir: Path, plots_dir: Path) -> list:
    """Macro-F1 / Micro-F1 vs. epoch -- the Task 1 deliverable, for every run."""
    written = []
    for hist_path in sorted(results_dir.glob("history_*.json")):
        history = load_json(hist_path)
        if not history:
            continue
        tag = hist_path.stem.replace("history_", "")
        epochs = [h["epoch"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(epochs, [h["train"]["loss"] for h in history], "o-", label="train loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title(f"{tag}: training loss")
        ax1.grid(alpha=0.3); ax1.legend()

        plotted = False
        for key, style in (("macro_f1", "o-"), ("micro_f1", "s--"), ("mean_R@5", "^-")):
            series = [h["val"].get(key) for h in history]
            if any(isinstance(v, float) and np.isfinite(v) for v in series):
                ax2.plot(epochs, series, style, label=f"val {key}")
                plotted = True
        if plotted:
            ax2.set_xlabel("epoch"); ax2.set_ylabel("score")
            ax2.set_title(f"{tag}: validation metrics")
            ax2.grid(alpha=0.3); ax2.legend()
            ax2.set_ylim(0, 1.02)

        fig.tight_layout()
        out = plots_dir / f"curves_{tag}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        written.append(out)
    return written


def _dominant_tag_labels(track_ids, processed_dir: Path, top_n: int = 8):
    """Label each track by its rarest (most specific) tag, for t-SNE colouring.

    MagnaTagATune carries no genre or mood field, so the genre/mood panels the
    spec asks for are empty there and t-SNE was silently skipped. Tags are the
    label space actually being predicted, so colouring by tag answers the same
    question -- does the fused representation separate the classes? The rarest
    tag of each clip is used because the common ones ("rock", "quiet") apply to
    most of the corpus and colour everything the same.
    """
    index_path = Path(processed_dir) / "index.json"
    if not index_path.exists():
        return None
    index = {r["track_id"]: r.get("tags", []) for r in load_json(index_path)}

    freq: dict[str, int] = {}
    for tags in index.values():
        for t in tags:
            freq[t] = freq.get(t, 0) + 1
    if not freq:
        return None

    labels = []
    for tid in track_ids:
        tags = index.get(str(tid), [])
        labels.append(min(tags, key=lambda t: freq.get(t, 0)) if tags else "untagged")
    labels = np.array(labels)

    # Collapse the long tail so the legend stays readable.
    keep = {lab for lab, _ in sorted(
        ((l, int((labels == l).sum())) for l in set(labels)),
        key=lambda kv: -kv[1])[:top_n]}
    return np.array([l if l in keep else "other" for l in labels])


def plot_tsne(results_dir: Path, plots_dir: Path, seed: int = 42,
              processed_dir: Path | None = None) -> list:
    """t-SNE of the fused vector z, coloured by genre and by mood (spec 4.3)."""
    from sklearn.manifold import TSNE

    written = []
    for npz_path in sorted(results_dir.glob("embeddings_task3_*.npz")):
        blob = np.load(npz_path, allow_pickle=True)
        z = blob["z"]
        if z.shape[0] < 6:
            print(f"  t-SNE skipped for {npz_path.name}: only {z.shape[0]} points")
            continue
        perplexity = float(min(30, max(5, (z.shape[0] - 1) / 3)))
        coords = TSNE(n_components=2, perplexity=perplexity, init="pca",
                      random_state=seed, max_iter=1000).fit_transform(z)

        panels = [(k, blob[k]) for k in ("genre", "mood")
                  if k in blob and len({str(v) for v in blob[k]}) > 1]
        if not panels and processed_dir is not None and "track_id" in blob:
            tag_labels = _dominant_tag_labels(blob["track_id"], processed_dir)
            if tag_labels is not None and len(set(tag_labels)) > 1:
                panels = [("most specific tag", tag_labels)]
                print("  t-SNE: no genre/mood on this corpus -- colouring by tag")
        if not panels:
            print(f"  t-SNE skipped for {npz_path.name}: nothing to colour by")
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5),
                                 squeeze=False)
        for ax, (name, labels) in zip(axes[0], panels):
            labels = np.array([str(v) if str(v) else "unlabelled" for v in labels])
            for value in sorted(set(labels)):
                mask = labels == value
                ax.scatter(coords[mask, 0], coords[mask, 1], s=28, alpha=0.8, label=value)
            ax.set_title(f"t-SNE of fused z by {name}")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=8, markerscale=0.8, loc="best")
        fig.tight_layout()
        out = plots_dir / f"tsne_{npz_path.stem.replace('embeddings_', '')}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        written.append(out)
    return written


def plot_comparison(rows: dict, plots_dir: Path) -> Path | None:
    """Grouped bar chart of Macro-F1 / Micro-F1 / AUC-PR across models."""
    keys = ["macro_f1", "micro_f1", "auc_pr"]
    names = [n for n, m in rows.items()
             if any(isinstance(m.get(k), float) and np.isfinite(m[k]) for k in keys)]
    if not names:
        return None

    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(9, 1.15 * len(names)), 5))
    for i, key in enumerate(keys):
        vals = [rows[n].get(key) if isinstance(rows[n].get(key), float)
                and np.isfinite(rows[n].get(key)) else 0.0 for n in names]
        ax.bar(x + (i - 1) * width, vals, width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
    ax.set_title("Model comparison (test split)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    out = plots_dir / "model_comparison.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_pr_curves(results_dir: Path, plots_dir: Path, processed_dir: Path,
                   top_n: int = 8) -> Path | None:
    """Per-tag precision-recall curves for the strongest available model."""
    from sklearn.metrics import precision_recall_curve

    preferred = ["predictions_task3_fusion_cross_attention.npz",
                 "predictions_task2_gnn_sage.npz", "predictions_task1_bert.npz"]
    path = next((results_dir / p for p in preferred if (results_dir / p).exists()), None)
    if path is None:
        return None

    blob = np.load(path, allow_pickle=True)
    scores, targets = blob["scores"], blob["targets"]
    label_space = load_json(Path(processed_dir) / "label_space.json")
    tag_names = label_space["tags"]

    support = targets.sum(axis=0)
    order = [k for k in np.argsort(-support) if 0 < support[k] < targets.shape[0]][:top_n]
    if not order:
        return None

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for k in order:
        precision, recall, _ = precision_recall_curve(targets[:, k], scores[:, k])
        name = tag_names[k] if k < len(tag_names) else f"tag{k}"
        ax.plot(recall, precision, label=f"{name} (n={int(support[k])})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"Per-tag PR curves -- {path.stem.replace('predictions_', '')}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    out = plots_dir / "pr_curves.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_ablation(metrics: dict, plots_dir: Path) -> Path | None:
    """The Task 3 ablation as a single bar chart."""
    ablation = metrics.get("task3_ablation")
    if not ablation:
        return None
    modes = list(ablation)
    vals = [ablation[m].get("macro_f1") or 0.0 for m in modes]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(modes, vals, color=["#888", "#888", "#5b8", "#38a"][: len(modes)])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylabel("Macro-F1 (test)"); ax.set_ylim(0, max(vals + [0.1]) * 1.25)
    ax.set_title("Task 3 fusion ablation")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    out = plots_dir / "ablation_task3.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate metrics and render plots.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    results_dir = resolve(cfg, "results")
    plots_dir = resolve(cfg, "plots")

    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("No results/metrics.json -- run src/train.py first.")
    metrics = load_json(metrics_path)

    table, rows = build_table(metrics)
    print("\n" + table + "\n")
    (results_dir / "comparison_table.txt").write_text(table, encoding="utf-8")
    save_json(rows, results_dir / "comparison_table.json")

    if args.table_only:
        return 0

    written = []
    written += plot_training_curves(results_dir, plots_dir)
    for fn in (lambda: plot_comparison(rows, plots_dir),
               lambda: plot_ablation(metrics, plots_dir),
               lambda: plot_pr_curves(results_dir, plots_dir,
                                      resolve(cfg, 'processed'))):
        try:
            p = fn()
            if p:
                written.append(p)
        except Exception as exc:  # noqa: BLE001 - a missing artefact must not abort
            print(f"  plot skipped: {type(exc).__name__}: {exc}")
    try:
        written += plot_tsne(results_dir, plots_dir, cfg.get("seed", 42),
                            resolve(cfg, "processed"))
    except Exception as exc:  # noqa: BLE001
        print(f"  t-SNE skipped: {type(exc).__name__}: {exc}")

    print(f"wrote {len(written)} plots -> {plots_dir}")
    for p in written:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
