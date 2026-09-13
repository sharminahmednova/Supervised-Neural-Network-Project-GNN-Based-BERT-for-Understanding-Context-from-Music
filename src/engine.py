"""Shared training / evaluation loop for every task.

One `Trainer` drives all four tasks. A task supplies three callables:

    forward(model, batch)  -> dict with "loss" and whatever the task predicts
    evaluate(model, loader)-> metrics dict for a split
    monitor                -> key in that dict to early-stop on (higher is better)

Everything else -- early stopping, best-checkpoint restore, discriminative
learning rates for BERT, gradient clipping, history for the F1-vs-epoch curves
the spec asks for -- is shared.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils import AverageMeter, count_parameters, save_json


@dataclass
class TrainState:
    best_score: float = -float("inf")
    best_epoch: int = -1
    best_state: dict | None = None
    epochs_without_improvement: int = 0
    history: list = field(default_factory=list)


def build_optimizer(model: nn.Module, lr: float, bert_lr: float,
                    weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Discriminative LRs: pretrained BERT weights move far slower than new heads.

    Using one LR for both wrecks the pretrained encoder in the first few steps
    on a small corpus, which is exactly where this project lives.
    """
    bert_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (bert_params if "bert" in name.lower() else other_params).append(param)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": lr})
    if bert_params:
        groups.append({"params": bert_params, "lr": bert_lr})
    if not groups:
        raise RuntimeError("model has no trainable parameters")
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


class Trainer:
    def __init__(self, model, forward_fn, evaluate_fn, cfg, device,
                 monitor: str = "macro_f1", tag: str = "run",
                 checkpoint_dir: Path | None = None, verbose: bool = True,
                 provenance: dict | None = None):
        self.provenance = provenance
        self.model = model.to(device)
        self.forward_fn = forward_fn
        self.evaluate_fn = evaluate_fn
        self.cfg = cfg
        self.device = device
        self.monitor = monitor
        self.tag = tag
        self.verbose = verbose
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        t = cfg["train"]
        self.epochs = t["epochs"]
        self.grad_clip = t.get("grad_clip", 0.0)
        self.patience = t.get("patience", 5)
        self.optimizer = build_optimizer(
            self.model, t["lr"], t.get("bert_lr", t["lr"]), t.get("weight_decay", 0.0))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=max(self.patience // 2, 1))
        self.state = TrainState()

    # -- one epoch ---------------------------------------------------------- #
    def train_epoch(self, loader) -> dict:
        self.model.train()
        meter = AverageMeter()
        parts_sum: dict = {}
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.forward_fn(self.model, batch)
            loss = out["loss"]
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.grad_clip)
            self.optimizer.step()

            n = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1
            meter.update(float(loss.detach()), n)
            for k, v in (out.get("parts") or {}).items():
                parts_sum.setdefault(k, AverageMeter()).update(v, n)
        return {"loss": meter.avg, **{k: m.avg for k, m in parts_sum.items()}}

    # -- full run ----------------------------------------------------------- #
    def fit(self, train_loader, val_loader) -> TrainState:
        total, trainable = count_parameters(self.model)
        if self.verbose:
            print(f"[{self.tag}] params: {total/1e6:.2f}M total, "
                  f"{trainable/1e6:.2f}M trainable | device={self.device}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_stats = self.train_epoch(train_loader)
            val_metrics = self.evaluate_fn(self.model, val_loader)
            score = val_metrics.get(self.monitor, -float("inf"))
            if not np.isfinite(score):
                score = -float("inf")
            self.scheduler.step(score)

            entry = {"epoch": epoch, "train": train_stats, "val": val_metrics,
                     "seconds": round(time.time() - t0, 2),
                     "lr": self.optimizer.param_groups[0]["lr"]}
            self.state.history.append(entry)

            improved = score > self.state.best_score + 1e-6
            if improved:
                self.state.best_score = score
                self.state.best_epoch = epoch
                self.state.best_state = copy.deepcopy(
                    {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()})
                self.state.epochs_without_improvement = 0
            else:
                self.state.epochs_without_improvement += 1

            if self.verbose:
                extras = "  ".join(
                    f"val_{k}={val_metrics[k]:.4f}"
                    for k in (self.monitor, "micro_f1", "auc_pr", "mae_mean", "mean_R@5")
                    if isinstance(val_metrics.get(k), float) and np.isfinite(val_metrics[k]))
                print(f"  epoch {epoch:>3}/{self.epochs}  "
                      f"loss={train_stats['loss']:.4f}  {extras}  "
                      f"({entry['seconds']:.1f}s){'  *' if improved else ''}")

            if self.state.epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best {self.monitor}={self.state.best_score:.4f} "
                          f"@ epoch {self.state.best_epoch})")
                break

        self.restore_best()
        return self.state

    def restore_best(self) -> None:
        if self.state.best_state is not None:
            self.model.load_state_dict(self.state.best_state)
            self.model.to(self.device)

    def save(self, path: Path | None = None, provenance: dict | None = None) -> Path | None:
        path = path or (self.checkpoint_dir / f"{self.tag}.pt" if self.checkpoint_dir else None)
        if path is None:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Stamp the corpus fingerprint into the checkpoint. Without it a
        # checkpoint trained on one corpus loads against another and fails deep
        # inside load_state_dict with a bare shape mismatch -- which is exactly
        # what happened when synthetic (32-tag) weights met GTZAN (10-tag) data.
        torch.save({"state_dict": self.model.state_dict(),
                    "best_score": self.state.best_score,
                    "best_epoch": self.state.best_epoch,
                    "monitor": self.monitor,
                    "tag": self.tag,
                    "provenance": provenance or self.provenance}, path)
        return path

    def save_history(self, path: Path) -> None:
        save_json(self.state.history, path)


# --------------------------------------------------------------------------- #
# Prediction collection
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictions(model, loader, device, forward_scores) -> dict:
    """Run a split and gather tag scores, tag targets, emotion, and ids.

    `forward_scores(model, batch)` returns (tag_logits, emotion_or_None).
    """
    model.eval()
    scores, targets, emo_pred, emo_true, emo_mask, ids, texts, genres = \
        [], [], [], [], [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits, emotion = forward_scores(model, batch)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch.y.float().cpu().numpy())
        if emotion is not None:
            emo_pred.append(emotion.cpu().numpy())
            emo_true.append(batch.va.cpu().numpy())
            emo_mask.append(batch.va_mask.cpu().numpy())
        ids.extend(_as_list(batch.track_id))
        texts.extend(_as_list(getattr(batch, "text", [])))
        genres.extend(_as_list(getattr(batch, "genre", [])))

    out = {
        "scores": np.concatenate(scores) if scores else np.zeros((0, 0)),
        "targets": np.concatenate(targets) if targets else np.zeros((0, 0)),
        "track_ids": ids,
        "texts": texts,
        "genres": genres,
    }
    if emo_pred:
        out["emotion_pred"] = np.concatenate(emo_pred)
        out["emotion_true"] = np.concatenate(emo_true)
        out["emotion_mask"] = np.concatenate(emo_mask).reshape(-1)
    return out


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def emotion_stats(dataset) -> tuple:
    """Mean/std of (valence, arousal) over annotated training tracks.

    Targets on the DEAM 1-9 scale would otherwise dominate the BCE term by two
    orders of magnitude; the loss standardises with these and the evaluator
    inverts them so reported MAE stays on the original scale.
    """
    rows = [dataset[i].va.reshape(-1).numpy() for i in range(len(dataset))
            if float(dataset[i].va_mask.reshape(-1)[0]) > 0.5]
    if not rows:
        return (torch.zeros(2), torch.ones(2))
    arr = np.stack(rows)
    mean = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(np.maximum(arr.std(axis=0), 1e-3), dtype=torch.float32)
    return mean, std


class CheckpointMismatch(RuntimeError):
    """Raised when a checkpoint was trained on a different corpus than the data."""


def load_checkpoint(path, dataset=None, map_location="cpu") -> dict:
    """Load a checkpoint, refusing one trained on a different label space.

    `load_state_dict` would otherwise fail with a bare tensor-shape error that
    says nothing about the cause. Pass the dataset you intend to run against and
    this explains the mismatch instead.
    """
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    prov = blob.get("provenance") or {}

    if dataset is not None:
        want_tags = len(dataset.label_space)
        got_tags = prov.get("num_tags")
        want_corpus = dataset.meta.get("source")
        got_corpus = prov.get("corpus")

        if got_tags is not None and got_tags != want_tags:
            raise CheckpointMismatch(
                f"{Path(path).name} was trained on {got_corpus!r} with "
                f"{got_tags} tags, but the processed data has {want_tags} tags "
                f"({want_corpus!r}). Re-run training for this corpus, or point "
                f"paths.processed at the corpus the checkpoint belongs to.")
        if got_corpus and want_corpus and got_corpus != want_corpus:
            raise CheckpointMismatch(
                f"{Path(path).name} was trained on {got_corpus!r} but the "
                f"processed data is {want_corpus!r}.")
        if got_tags is None:
            print(f"  warning: {Path(path).name} carries no provenance "
                  f"(pre-dates checkpoint stamping); compatibility unchecked.")
    return blob
