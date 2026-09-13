"""Evaluation metrics (spec section 6).

Tagging   : per-tag precision/recall/F1, Macro-F1, Micro-F1, mean AUC-PR
Emotion   : MAE and R^2 for valence/arousal
Retrieval : Recall@K in both directions, plus median rank and MRR
Analysis  : graph coherence (see gnn_model.graph_coherence_score)

Tags that never occur in a split are excluded from the macro average -- F1 is
undefined for them, and averaging in a zero silently deflates every number.
`macro_f1_all_tags` keeps the zero-filled variant for comparability when a
report needs a fixed denominator.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_array(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Multi-label tagging
# --------------------------------------------------------------------------- #
def per_tag_prf(y_true, y_pred) -> dict:
    """Precision / recall / F1 per tag from binarised predictions."""
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "support": y_true.sum(axis=0)}


def tagging_metrics(y_true, y_scores, threshold: float = 0.5,
                    tag_names: list[str] | None = None) -> dict:
    """Macro/Micro-F1 and mean AUC-PR over tags.

    `y_scores` are probabilities in [0, 1]; `threshold` binarises them.
    """
    y_true = _as_array(y_true)
    y_scores = _as_array(y_scores)
    y_pred = (y_scores >= threshold).astype(np.float64)

    prf = per_tag_prf(y_true, y_pred)
    present = prf["support"] > 0

    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    micro_p = tp / (tp + fp) if tp + fp > 0 else 0.0
    micro_r = tp / (tp + fn) if tp + fn > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p + micro_r > 0 else 0.0)

    # AUC-PR / ROC-AUC need both classes present in the column.
    aucpr, rocauc = [], []
    for k in range(y_true.shape[1]):
        col = y_true[:, k]
        if 0 < col.sum() < len(col):
            aucpr.append(average_precision_score(col, y_scores[:, k]))
            try:
                rocauc.append(roc_auc_score(col, y_scores[:, k]))
            except ValueError:
                pass

    out = {
        "macro_f1": float(prf["f1"][present].mean()) if present.any() else 0.0,
        "macro_f1_all_tags": float(prf["f1"].mean()),
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "macro_precision": float(prf["precision"][present].mean()) if present.any() else 0.0,
        "macro_recall": float(prf["recall"][present].mean()) if present.any() else 0.0,
        "auc_pr": float(np.mean(aucpr)) if aucpr else float("nan"),
        "roc_auc": float(np.mean(rocauc)) if rocauc else float("nan"),
        "n_tags_scored": int(present.sum()),
        "n_samples": int(y_true.shape[0]),
        "threshold": float(threshold),
    }
    if tag_names is not None:
        out["per_tag"] = {
            name: {"precision": float(prf["precision"][k]),
                   "recall": float(prf["recall"][k]),
                   "f1": float(prf["f1"][k]),
                   "support": int(prf["support"][k])}
            for k, name in enumerate(tag_names)
        }
    return out


def tune_threshold(y_true, y_scores, grid=None) -> tuple[float, float]:
    """Pick the global threshold maximising Macro-F1 on a validation split.

    A single 0.5 cut is a poor operating point for sparse multi-label targets,
    so the trainers tune this on val and reuse it on test.
    """
    grid = grid if grid is not None else np.arange(0.05, 0.95, 0.05)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = tagging_metrics(y_true, y_scores, threshold=float(t))["macro_f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


# --------------------------------------------------------------------------- #
# Emotion regression (DEAM)
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred, mask=None, names=("valence", "arousal")) -> dict:
    """MAE, RMSE and R^2 per target; `mask` selects rows with real annotations."""
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    if mask is not None:
        keep = _as_array(mask).reshape(-1) > 0.5
        y_true, y_pred = y_true[keep], y_pred[keep]
    if y_true.size == 0:
        return {"n_samples": 0}

    out = {"n_samples": int(y_true.shape[0])}
    for i, name in enumerate(names[: y_true.shape[1]]):
        t, p = y_true[:, i], y_pred[:, i]
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        out[f"mae_{name}"] = float(np.abs(t - p).mean())
        out[f"rmse_{name}"] = float(np.sqrt(((t - p) ** 2).mean()))
        out[f"r2_{name}"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    maes = [out[f"mae_{n}"] for n in names[: y_true.shape[1]]]
    out["mae_mean"] = float(np.mean(maes))
    return out


# --------------------------------------------------------------------------- #
# Cross-modal retrieval (Task 4)
# --------------------------------------------------------------------------- #
def retrieval_metrics(sim: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Recall@K / median rank / MRR in both directions.

    `sim[i, j]` scores graph i against caption j; the diagonal is the true pair.

    Ties are resolved in EXPECTATION, not optimistically. Counting only
    strictly better candidates (`sim > diag`) gives every member of a tied group
    rank 1, which is not a rounding detail here: MagnaTagATune's text is its tag
    set, so after the disjoint tag partition ~35% of records share one identical
    placeholder string. Their similarities tie exactly, and optimistic ranking
    reported R@1 = 0.136 for a model that had collapsed to retrieving the same
    three tracks for every query.

    So for an item with `b` candidates strictly above it and `e` tied with it,
    we use the probability that random tie-breaking puts it in the top k,

        P(rank <= k) = clip((k - b) / (e + 1), 0, 1),

    which is the expected recall. A fully collapsed model then scores exactly
    chance (k/n), a perfectly separated one scores 1.0, and `median_rank` and
    MRR use the mid-rank b + 1 + e/2.
    """
    sim = _as_array(sim)
    n = sim.shape[0]
    out = {"n_pairs": int(n)}

    for direction, matrix in (("audio2caption", sim), ("caption2audio", sim.T)):
        diag = np.diag(matrix).reshape(-1, 1)
        better = (matrix > diag).sum(axis=1)
        # `equal` counts the true pair itself, hence the -1.
        equal = (matrix == diag).sum(axis=1) - 1
        mid_ranks = better + 1 + equal / 2.0

        for k in ks:
            expected = np.clip((k - better) / (equal + 1.0), 0.0, 1.0)
            out[f"{direction}_R@{k}"] = float(expected.mean())
        out[f"{direction}_median_rank"] = float(np.median(mid_ranks))
        out[f"{direction}_mrr"] = float((1.0 / mid_ranks).mean())
        out[f"{direction}_mean_ties"] = float(equal.mean())

    for k in ks:
        out[f"mean_R@{k}"] = float(
            (out[f"audio2caption_R@{k}"] + out[f"caption2audio_R@{k}"]) / 2)

    # Chance R@K, so a reader can tell a real result from a collapsed encoder.
    for k in ks:
        out[f"chance_R@{k}"] = float(min(k, n) / n) if n else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def format_metrics(metrics: dict, keys=None, precision: int = 4) -> str:
    """One-line summary for training logs."""
    keys = keys or ["macro_f1", "micro_f1", "auc_pr"]
    parts = []
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(f"{k}={v:.{precision}f}")
    return "  ".join(parts)


def comparison_table(rows: dict, columns=None) -> str:
    """Render the spec's model-comparison table (section 8) as fixed-width text."""
    columns = columns or ["macro_f1", "micro_f1", "auc_pr", "mae_mean", "mean_R@5"]
    name_w = max([len("Model")] + [len(k) for k in rows])
    header = "Model".ljust(name_w) + "".join(c.rjust(12) for c in columns)
    lines = [header, "-" * len(header)]
    for name, m in rows.items():
        cells = []
        for c in columns:
            v = m.get(c)
            cells.append(f"{v:.4f}".rjust(12) if isinstance(v, (int, float))
                         and not isinstance(v, bool) and np.isfinite(v)
                         else "-".rjust(12))
        lines.append(name.ljust(name_w) + "".join(cells))
    return "\n".join(lines)
