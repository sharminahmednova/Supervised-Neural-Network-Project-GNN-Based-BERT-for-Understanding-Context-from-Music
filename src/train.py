"""Training entry point for all four tasks and the baselines.

    python src/train.py task1                      # BERT tag classifier
    python src/train.py task2                      # GNN on structure graphs
    python src/train.py task3                      # GNN-BERT fusion
    python src/train.py task3 --ablation           # all four fusion arms
    python src/train.py task4                      # contrastive dual encoder
    python src/train.py baselines                  # B1, B2, B4
    python src/train.py all                        # everything, in order

Any config key can be overridden: `--set train.epochs=30 --set gnn.conv=gat`.
Per-run metrics accumulate in results/metrics.json; history (for the F1-vs-epoch
curves) lands in results/history_<tag>.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from baselines import MelCNN, PCAMLPBaseline, PriorBaseline, dense_batch  # noqa: E402
from bert_encoder import BertTagClassifier  # noqa: E402
from contrastive import (ContrastiveGNNBert, embed_split, info_nce_loss,  # noqa: E402
                         retrieval_examples, similarity_matrix, zero_shot_tag_scores)
from data.dataset import MusicContextDataset, make_loader  # noqa: E402
from engine import (Trainer, collect_predictions, emotion_stats)  # noqa: E402
from fusion_model import FUSION_MODES, MultiTaskLoss, build_fusion_model  # noqa: E402
from gnn_model import build_gnn, graph_coherence_score  # noqa: E402
from metrics import (format_metrics, regression_metrics, retrieval_metrics,  # noqa: E402
                     tagging_metrics, tune_threshold)
from utils import (REPO_ROOT, build_provenance, get_device, load_config,  # noqa: E402
                   load_json, merge_metrics, resolve, save_json, set_seed)


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def load_splits(cfg):
    processed, splits = resolve(cfg, "processed"), resolve(cfg, "splits")
    if not (processed / "meta.json").exists():
        raise SystemExit("No preprocessed data found. Run:  python src/preprocess.py")
    data = {s: MusicContextDataset(processed, splits, s) for s in ("train", "val", "test")}
    bs, nw = cfg["train"]["batch_size"], cfg["train"].get("num_workers", 0)
    loaders = {
        "train": make_loader(data["train"], bs, True, nw),
        "val": make_loader(data["val"], bs, False, nw),
        "test": make_loader(data["test"], bs, False, nw),
    }
    return data, loaders


def make_tag_evaluator(forward_scores, device, threshold_ref: dict,
                       emo_stats=None, tag_names=None):
    """Build an evaluate_fn that tunes its threshold on the split it is given."""

    def evaluate(model, loader) -> dict:
        preds = collect_predictions(model, loader, device, forward_scores)
        if preds["targets"].size == 0:
            return {"macro_f1": 0.0}
        threshold, _ = tune_threshold(preds["targets"], preds["scores"])
        threshold_ref["value"] = threshold
        out = tagging_metrics(preds["targets"], preds["scores"], threshold, tag_names)
        if "emotion_pred" in preds and preds["emotion_mask"].sum() > 0:
            pred = preds["emotion_pred"]
            if emo_stats is not None:              # undo standardisation
                mean, std = emo_stats
                pred = pred * std.numpy() + mean.numpy()
            out.update(regression_metrics(preds["emotion_true"], pred,
                                          preds["emotion_mask"]))
        return out

    return evaluate


def provenance_of(cfg, data) -> dict:
    """Corpus fingerprint for every metrics row this run writes."""
    return build_provenance(cfg, data["train"].meta)


def finalize(tag: str, cfg, trainer, forward_scores, loaders, device,
             threshold_ref: dict, emo_stats=None, tag_names=None,
             extra: dict | None = None, provenance: dict | None = None) -> dict:
    """Tune the threshold on val, score test at that threshold, and persist."""
    results_dir = resolve(cfg, "results")

    val_preds = collect_predictions(trainer.model, loaders["val"], device, forward_scores)
    threshold, val_f1 = tune_threshold(val_preds["targets"], val_preds["scores"])

    test_preds = collect_predictions(trainer.model, loaders["test"], device, forward_scores)
    test_metrics = tagging_metrics(test_preds["targets"], test_preds["scores"],
                                   threshold, tag_names)
    if "emotion_pred" in test_preds and test_preds["emotion_mask"].sum() > 0:
        pred = test_preds["emotion_pred"]
        if emo_stats is not None:
            mean, std = emo_stats
            pred = pred * std.numpy() + mean.numpy()
        test_metrics.update(regression_metrics(test_preds["emotion_true"], pred,
                                               test_preds["emotion_mask"]))

    payload = {
        "tag": tag,
        "split": "test",
        "threshold": threshold,
        "val_macro_f1_at_threshold": val_f1,
        "best_val_score": trainer.state.best_score,
        "best_epoch": trainer.state.best_epoch,
        **{k: v for k, v in test_metrics.items() if k != "per_tag"},
        **(extra or {}),
        "_provenance": provenance,
    }
    if "per_tag" in test_metrics:
        save_json(test_metrics["per_tag"], results_dir / f"per_tag_{tag}.json")

    trainer.save_history(results_dir / f"history_{tag}.json")
    trainer.save(results_dir / "checkpoints" / f"{tag}.pt", provenance)
    merge_metrics(results_dir / "metrics.json", tag, payload, provenance)
    np.savez_compressed(results_dir / f"predictions_{tag}.npz",
                        scores=test_preds["scores"], targets=test_preds["targets"],
                        track_ids=np.array(test_preds["track_ids"], dtype=object))

    print(f"\n[{tag}] TEST  {format_metrics(test_metrics)}  (threshold={threshold:.2f})")
    return payload


# --------------------------------------------------------------------------- #
# Task 1 -- BERT tag classifier
# --------------------------------------------------------------------------- #
def run_task1(cfg, device) -> dict:
    print("\n" + "=" * 72 + "\nTASK 1 (Easy): BERT multi-label tag classifier\n" + "=" * 72)
    data, loaders = load_splits(cfg)
    tags = data["train"].label_space.tags

    model = BertTagClassifier(
        num_tags=len(tags),
        model_name=cfg["text"]["model_name"],
        freeze=cfg["text"].get("freeze_bert", False),
        unfreeze_last_n=cfg["text"].get("unfreeze_last_n", 0),
        pooling="mean" if cfg["text"].get("freeze_bert", False) else "cls",
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=data["train"].pos_weight().to(device))

    def forward(m, batch):
        logits = m(batch.input_ids, batch.attention_mask)
        loss = criterion(logits, batch.y.float())
        return {"loss": loss, "parts": {"bce": float(loss.detach())}}

    def forward_scores(m, batch):
        return m(batch.input_ids, batch.attention_mask), None

    threshold_ref = {"value": 0.5}
    trainer = Trainer(model, forward, make_tag_evaluator(forward_scores, device, threshold_ref),
                      cfg, device, monitor="macro_f1", tag="task1_bert")
    trainer.fit(loaders["train"], loaders["val"])
    payload = finalize("task1_bert", cfg, trainer, forward_scores, loaders, device,
                       threshold_ref, tag_names=tags, provenance=provenance_of(cfg, data))
    export_task1_examples(cfg, trainer.model, data["test"], loaders["test"], device, tags)
    return payload


@torch.no_grad()
def export_task1_examples(cfg, model, dataset, loader, device, tags, n: int = 5) -> None:
    """Deliverable: 5 example predictions with token-saliency highlighting."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model_name"])
    model.eval()
    examples = []
    for batch in loader:
        batch = batch.to(device)
        logits, saliency = model(batch.input_ids, batch.attention_mask, return_attention=True)
        probs = torch.sigmoid(logits)
        for i in range(probs.size(0)):
            if len(examples) >= n:
                break
            order = saliency[i].topk(min(8, saliency.size(1))).indices
            toks = tokenizer.convert_ids_to_tokens(batch.input_ids[i][order].tolist())
            true = [tags[k] for k, v in enumerate(batch.y[i].tolist()) if v > 0.5]
            top = probs[i].topk(min(5, probs.size(1)))
            examples.append({
                "track_id": batch.track_id[i] if isinstance(batch.track_id, list) else "?",
                "text": batch.text[i] if isinstance(batch.text, list) else "",
                "true_tags": true,
                "predicted_top5": [{"tag": tags[j], "score": round(float(s), 4)}
                                   for s, j in zip(top.values.tolist(), top.indices.tolist())],
                "salient_tokens": [t for t in toks if t not in ("[PAD]",)],
            })
        if len(examples) >= n:
            break
    out_path = resolve(cfg, "results") / "task1_examples.json"
    save_json(examples, out_path)
    print(f"  wrote {len(examples)} example predictions -> {out_path}")


# --------------------------------------------------------------------------- #
# Task 2 -- GNN on music structure graphs
# --------------------------------------------------------------------------- #
def run_task2(cfg, device) -> dict:
    print("\n" + "=" * 72 + "\nTASK 2 (Medium): GNN on music structure graphs\n" + "=" * 72)
    data, loaders = load_splits(cfg)
    tags = data["train"].label_space.tags

    model = build_gnn(cfg, data["train"].node_feature_dim, len(tags))
    criterion = nn.BCEWithLogitsLoss(pos_weight=data["train"].pos_weight().to(device))

    def forward(m, batch):
        loss = criterion(m(batch), batch.y.float())
        return {"loss": loss, "parts": {"bce": float(loss.detach())}}

    def forward_scores(m, batch):
        return m(batch), None

    threshold_ref = {"value": 0.5}
    tag_name = f"task2_gnn_{cfg['gnn']['conv']}"
    trainer = Trainer(model, forward, make_tag_evaluator(forward_scores, device, threshold_ref),
                      cfg, device, monitor="macro_f1", tag=tag_name)
    trainer.fit(loaders["train"], loaders["val"])

    # Report S_graph alongside its lift over non-edges: S_graph alone saturates
    # (ReLU features are non-negative), so the lift is what carries the signal.
    parts = [graph_coherence_score(trainer.model.encoder, b.to(device), detailed=True)
             for b in loaders["test"]]
    extra = {}
    for key in ("s_graph", "edge_similarity", "non_edge_similarity", "coherence_lift"):
        vals = [p[key] for p in parts if key in p and np.isfinite(p[key])]
        if vals:
            extra[f"graph_{key}" if key != "s_graph" else "graph_coherence"] = \
                float(np.mean(vals))
    return finalize(tag_name, cfg, trainer, forward_scores, loaders, device,
                    threshold_ref, tag_names=tags, extra=extra,
                    provenance=provenance_of(cfg, data))


# --------------------------------------------------------------------------- #
# Task 3 -- GNN-BERT fusion
# --------------------------------------------------------------------------- #
def run_task3(cfg, device, mode: str | None = None) -> dict:
    mode = mode or cfg["fusion"].get("mode", "cross_attention")
    print("\n" + "=" * 72 + f"\nTASK 3 (Hard): GNN-BERT fusion [{mode}]\n" + "=" * 72)
    cfg["fusion"]["mode"] = mode

    data, loaders = load_splits(cfg)
    tags = data["train"].label_space.tags
    predict_emotion = bool(data["train"].meta.get("has_emotion"))
    stats = emotion_stats(data["train"]) if predict_emotion else None

    model = build_fusion_model(cfg, data["train"].node_feature_dim, len(tags), predict_emotion)
    criterion = MultiTaskLoss(cfg["train"].get("alpha_valence", 0.3),
                              cfg["train"].get("beta_arousal", 0.3),
                              data["train"].pos_weight().to(device))

    def forward(m, batch):
        out = m(batch)
        loss, parts = criterion(out, batch, stats)
        return {"loss": loss, "parts": parts}

    def forward_scores(m, batch):
        out = m(batch)
        return out["tag_logits"], out.get("emotion")

    threshold_ref = {"value": 0.5}
    tag_name = f"task3_fusion_{mode}"
    trainer = Trainer(model, forward,
                      make_tag_evaluator(forward_scores, device, threshold_ref, stats),
                      cfg, device, monitor="macro_f1", tag=tag_name)
    trainer.fit(loaders["train"], loaders["val"])
    payload = finalize(tag_name, cfg, trainer, forward_scores, loaders, device,
                       threshold_ref, stats, tags, extra={"fusion_mode": mode},
                       provenance=provenance_of(cfg, data))

    if mode == "cross_attention":
        export_embeddings(cfg, trainer.model, loaders, device, tag_name)
        export_case_studies(cfg, trainer.model, loaders["test"], device, tags)
    return payload


def run_task3_ablation(cfg, device) -> dict:
    """The spec's ablation: BERT-only / GNN-only / early concat / cross-attention."""
    results = {}
    for mode in FUSION_MODES:
        results[mode] = run_task3(cfg, device, mode)
    merge_metrics(resolve(cfg, "results") / "metrics.json", "task3_ablation",
                  {m: {k: r.get(k) for k in ("macro_f1", "micro_f1", "auc_pr", "mae_mean")}
                   for m, r in results.items()},
                  next(iter(results.values()), {}).get("_provenance")
                  or build_provenance(cfg))
    print("\nAblation summary (test):")
    for mode, r in results.items():
        print(f"  {mode:16s} macro-F1={r.get('macro_f1', float('nan')):.4f}  "
              f"AUC-PR={r.get('auc_pr', float('nan')):.4f}")
    return results


@torch.no_grad()
def export_embeddings(cfg, model, loaders, device, tag_name: str) -> None:
    """Save fused vectors z (+ genre/mood labels) for the t-SNE deliverable."""
    model.eval()
    zs, genres, ids, moods = [], [], [], []
    index = {r["track_id"]: r
             for r in load_json(resolve(cfg, "processed") / "index.json")}
    for batch in loaders["test"]:
        batch = batch.to(device)
        out = model(batch, return_embedding=True)
        zs.append(out["z"].cpu().numpy())
        batch_ids = batch.track_id if isinstance(batch.track_id, list) else [batch.track_id]
        ids.extend(batch_ids)
        genres.extend(batch.genre if isinstance(batch.genre, list) else [batch.genre])
        moods.extend(index.get(t, {}).get("mood", "") for t in batch_ids)
    path = resolve(cfg, "results") / f"embeddings_{tag_name}.npz"
    np.savez_compressed(path, z=np.concatenate(zs),
                        genre=np.array(genres, dtype=object),
                        mood=np.array(moods, dtype=object),
                        track_id=np.array(ids, dtype=object))
    print(f"  wrote fused embeddings -> {path.name}")


@torch.no_grad()
def export_case_studies(cfg, model, loader, device, tags, n: int = 3) -> None:
    """Deliverable: graph paths aligned against the attended caption tokens."""
    from transformers import AutoTokenizer

    from fusion_model import top_attended_tokens
    from graph_builder import EDGE_TYPE_NAMES

    tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model_name"])
    model.eval()
    studies = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        attn = out["attention"]
        if attn is None:
            break
        probs = torch.sigmoid(out["tag_logits"])
        attended = top_attended_tokens(tokenizer, batch.input_ids, attn, k=6)

        for i in range(probs.size(0)):
            if len(studies) >= n:
                break
            node_mask = (batch.batch == i)
            edge_mask = node_mask[batch.edge_index[0]]
            etypes = batch.edge_type[edge_mask].tolist()
            offset = int(node_mask.nonzero()[0])
            edges = batch.edge_index[:, edge_mask].cpu().numpy() - offset
            path = [f"s{int(a)}->s{int(b)}" for a, b, t in
                    zip(edges[0], edges[1], etypes)
                    if t != 3 and int(a) != int(b)][:8]
            top = probs[i].topk(min(5, probs.size(1)))
            studies.append({
                "track_id": batch.track_id[i] if isinstance(batch.track_id, list) else "?",
                "caption": batch.text[i] if isinstance(batch.text, list) else "",
                "true_tags": [tags[k] for k, v in enumerate(batch.y[i].tolist()) if v > 0.5],
                "predicted_top5": [{"tag": tags[j], "score": round(float(s), 4)}
                                   for s, j in zip(top.values.tolist(), top.indices.tolist())],
                "graph_path": path,
                "edge_types_present": sorted({EDGE_TYPE_NAMES.get(t, str(t)) for t in etypes}),
                "top_attended_caption_tokens": attended[i] if i < len(attended) else [],
            })
        if len(studies) >= n:
            break
    out_path = resolve(cfg, "results") / "task3_case_studies.json"
    save_json(studies, out_path)
    print(f"  wrote {len(studies)} case studies -> {out_path}")


# --------------------------------------------------------------------------- #
# Task 4 -- contrastive dual encoder
# --------------------------------------------------------------------------- #
def run_task4(cfg, device) -> dict:
    print("\n" + "=" * 72 + "\nTASK 4 (Advanced): contrastive GNN-BERT retrieval\n" + "=" * 72)
    data, loaders = load_splits(cfg)
    tags = data["train"].label_space.tags

    if cfg["train"]["batch_size"] < 4:
        print("  note: InfoNCE needs in-batch negatives; batch_size < 4 is degenerate")

    model = ContrastiveGNNBert(data["train"].node_feature_dim, cfg)

    def forward(m, batch):
        out = m(batch)
        loss, parts = info_nce_loss(out["logits"])
        return {"loss": loss, "parts": parts}

    def evaluate(m, loader) -> dict:
        emb = embed_split(m, loader, device)
        sim = similarity_matrix(emb["graph"], emb["text"])
        return retrieval_metrics(sim)

    trainer = Trainer(model, forward, evaluate, cfg, device,
                      monitor="mean_R@5", tag="task4_contrastive")
    trainer.fit(loaders["train"], loaders["val"])

    results_dir = resolve(cfg, "results")
    emb = embed_split(trainer.model, loaders["test"], device)
    sim = similarity_matrix(emb["graph"], emb["text"])
    test_metrics = retrieval_metrics(sim)

    examples = retrieval_examples(sim, emb["captions"], emb["track_ids"], n=10, top_k=3)
    retrieval_path = resolve(cfg, "retrieval") / "caption_to_audio.json"
    save_json(examples, retrieval_path)
    np.savez_compressed(results_dir / "embeddings_task4.npz",
                        graph=emb["graph"], text=emb["text"],
                        track_id=np.array(emb["track_ids"], dtype=object))

    # Zero-shot tagging from captions, vs. the Task 3 supervised model.
    # Cosine scores live in [-1, 1]; rescale to [0, 1] using the *val* range so
    # the operating point -- like every other threshold here -- is never fitted
    # on test.
    zs_val_scores, zs_val_targets = zero_shot_tag_scores(
        trainer.model, loaders["val"], tags, device)
    lo, hi = float(zs_val_scores.min()), float(zs_val_scores.max())
    span = max(hi - lo, 1e-9)

    def _rescale(s):
        return np.clip((s - lo) / span, 0.0, 1.0)

    zs_threshold, _ = tune_threshold(zs_val_targets, _rescale(zs_val_scores))
    zs_scores, zs_targets = zero_shot_tag_scores(trainer.model, loaders["test"], tags, device)
    zs_metrics = tagging_metrics(zs_targets, _rescale(zs_scores), zs_threshold)

    payload = {"tag": "task4_contrastive", "split": "test", **test_metrics,
               "best_val_score": trainer.state.best_score,
               "best_epoch": trainer.state.best_epoch,
               "zero_shot_macro_f1": zs_metrics["macro_f1"],
               "zero_shot_micro_f1": zs_metrics["micro_f1"],
               "zero_shot_auc_pr": zs_metrics["auc_pr"]}
    trainer.save_history(results_dir / "history_task4_contrastive.json")
    trainer.save(results_dir / "checkpoints" / "task4_contrastive.pt",
                 provenance_of(cfg, data))
    merge_metrics(results_dir / "metrics.json", "task4_contrastive", payload,
                  provenance_of(cfg, data))

    print(f"\n[task4] TEST  R@1={test_metrics['mean_R@1']:.4f}  "
          f"R@5={test_metrics['mean_R@5']:.4f}  R@10={test_metrics['mean_R@10']:.4f}")
    print(f"  zero-shot tagging macro-F1={zs_metrics['macro_f1']:.4f} "
          f"(supervised Task 3 is the comparison point)")
    print(f"  wrote {len(examples)} retrieval examples -> {retrieval_path}")
    return payload


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def run_baselines(cfg, device) -> dict:
    print("\n" + "=" * 72 + "\nBASELINES: B1 prior/random, B2 mel-CNN, B4 PCA+MLP\n" + "=" * 72)
    data, loaders = load_splits(cfg)
    tags = data["train"].label_space.tags
    results_dir = resolve(cfg, "results")
    out = {}

    prov = provenance_of(cfg, data)
    y_train = data["train"].label_matrix()
    y_val = data["val"].label_matrix()
    y_test = data["test"].label_matrix()

    # -- B1 ---------------------------------------------------------------- #
    # Thresholds are tuned on val and only then applied to test -- tuning on
    # test would hand the baselines an advantage the trained models never get,
    # and a random predictor can score surprisingly well that way.
    for mode in ("random", "prior"):
        b1 = PriorBaseline(mode, cfg.get("seed", 42)).fit(y_train)
        t, _ = tune_threshold(y_val, b1.predict_proba(len(y_val)))
        scores = b1.predict_proba(len(y_test))
        m = tagging_metrics(y_test, scores, t)
        out[f"baseline_b1_{mode}"] = {"tag": f"baseline_b1_{mode}", "threshold": t, **m}
        merge_metrics(results_dir / "metrics.json", f"baseline_b1_{mode}",
                      {k: v for k, v in m.items() if k != "per_tag"}, prov)
        print(f"  B1 [{mode:6s}]  {format_metrics(m)}")

    # -- B2: CNN on the same node features, minus the graph ----------------- #
    max_nodes = max(int(data[s][i].num_nodes) for s in data for i in range(len(data[s])))
    cnn = MelCNN(data["train"].node_feature_dim, len(tags)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=data["train"].pos_weight().to(device))

    def forward(m, batch):
        loss = criterion(m(dense_batch(batch, max_nodes)), batch.y.float())
        return {"loss": loss, "parts": {"bce": float(loss.detach())}}

    def forward_scores(m, batch):
        return m(dense_batch(batch, max_nodes)), None

    threshold_ref = {"value": 0.5}
    trainer = Trainer(cnn, forward, make_tag_evaluator(forward_scores, device, threshold_ref),
                      cfg, device, monitor="macro_f1", tag="baseline_b2_melcnn")
    trainer.fit(loaders["train"], loaders["val"])
    out["baseline_b2_melcnn"] = finalize("baseline_b2_melcnn", cfg, trainer, forward_scores,
                                         loaders, device, threshold_ref, tag_names=tags,
                                         provenance=prov)

    # -- B4: PCA + MLP ------------------------------------------------------ #
    try:
        X_train, Y_train = PCAMLPBaseline.featurize(data["train"])
        X_val, Y_val = PCAMLPBaseline.featurize(data["val"])
        X_test, Y_test = PCAMLPBaseline.featurize(data["test"])
        b4 = PCAMLPBaseline(seed=cfg.get("seed", 42)).fit(X_train, Y_train)
        t, _ = tune_threshold(Y_val, b4.predict_proba(X_val))
        scores = b4.predict_proba(X_test)
        m = tagging_metrics(Y_test, scores, t)
        out["baseline_b4_pca_mlp"] = {"tag": "baseline_b4_pca_mlp", "threshold": t, **m}
        merge_metrics(results_dir / "metrics.json", "baseline_b4_pca_mlp",
                      {k: v for k, v in m.items() if k != "per_tag"}, prov)
        print(f"  B4 [pca+mlp]  {format_metrics(m)}")
    except Exception as exc:  # noqa: BLE001 - B4 is optional in the spec
        print(f"  B4 skipped: {type(exc).__name__}: {exc}")

    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Train GNN-BERT music context models.")
    ap.add_argument("task", choices=["task1", "task2", "task3", "task4",
                                     "baselines", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    ap.add_argument("--ablation", action="store_true",
                    help="task3: run all four fusion arms")
    ap.add_argument("--mode", default=None, choices=list(FUSION_MODES),
                    help="task3: a single fusion arm")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    if args.task == "task1":
        run_task1(cfg, device)
    elif args.task == "task2":
        run_task2(cfg, device)
    elif args.task == "task3":
        run_task3_ablation(cfg, device) if args.ablation else run_task3(cfg, device, args.mode)
    elif args.task == "task4":
        run_task4(cfg, device)
    elif args.task == "baselines":
        run_baselines(cfg, device)
    else:
        run_baselines(cfg, device)
        run_task1(cfg, device)
        run_task2(cfg, device)
        run_task3_ablation(cfg, device)
        run_task4(cfg, device)

    print(f"\nmetrics -> {resolve(cfg, 'results') / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
