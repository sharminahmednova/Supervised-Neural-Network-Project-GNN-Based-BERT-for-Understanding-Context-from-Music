"""Regenerate the graded export artifacts from a saved checkpoint.

The spec names several artifacts as deliverables in their own right:

    Task 1  5 example predictions with attention visualisation
    Task 3  t-SNE of the fused z (by genre and mood), 3 case studies
    Task 4  10 qualitative caption -> top-3 retrieval examples

Those used to be produced only as a side effect of `train.py`, so losing the
files meant retraining to get them back -- which is exactly what happened here.
This regenerates them from `results/checkpoints/*.pt` plus the processed data,
in seconds, with no training.

    python tools/export_deliverables.py --all
    python tools/export_deliverables.py --task task3 --mode cross_attention
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from audio_features import node_feature_dim  # noqa: E402
from data.dataset import MusicContextDataset, make_loader  # noqa: E402
from engine import load_checkpoint  # noqa: E402
from utils import get_device, load_config, resolve, set_seed  # noqa: E402


def _splits(cfg):
    processed, splits = resolve(cfg, "processed"), resolve(cfg, "splits")
    if not (processed / "meta.json").exists():
        raise SystemExit("No preprocessed data. Run: python src/preprocess.py")
    data = {s: MusicContextDataset(processed, splits, s) for s in ("train", "val", "test")}
    bs = cfg["train"]["batch_size"]
    loaders = {s: make_loader(data[s], bs, False) for s in data}
    return data, loaders


def _ckpt(cfg, name: str) -> Path:
    path = resolve(cfg, "results") / "checkpoints" / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}\n"
                         f"train it first, e.g.  python src/train.py {name.split('_')[0]}")
    return path


# --------------------------------------------------------------------------- #
def export_task1(cfg, device) -> None:
    from bert_encoder import BertTagClassifier
    from train import export_task1_examples

    data, loaders = _splits(cfg)
    tags = data["train"].label_space.tags
    blob = load_checkpoint(_ckpt(cfg, "task1_bert"), data["train"], device)

    model = BertTagClassifier(
        num_tags=len(tags),
        model_name=cfg["text"]["model_name"],
        freeze=cfg["text"].get("freeze_bert", False),
        unfreeze_last_n=cfg["text"].get("unfreeze_last_n", 0),
        pooling="mean" if cfg["text"].get("freeze_bert", False) else "cls",
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    export_task1_examples(cfg, model, data["test"], loaders["test"], device, tags)


def export_task3(cfg, device, mode: str = "cross_attention") -> None:
    from fusion_model import build_fusion_model
    from train import export_case_studies, export_embeddings

    cfg["fusion"]["mode"] = mode
    data, loaders = _splits(cfg)
    tags = data["train"].label_space.tags
    tag_name = f"task3_fusion_{mode}"
    blob = load_checkpoint(_ckpt(cfg, tag_name), data["train"], device)

    predict_emotion = bool(data["train"].meta.get("has_emotion"))
    model = build_fusion_model(cfg, data["train"].node_feature_dim,
                               len(tags), predict_emotion).to(device)
    model.load_state_dict(blob["state_dict"])

    export_embeddings(cfg, model, loaders, device, tag_name)
    if mode == "cross_attention":
        export_case_studies(cfg, model, loaders["test"], device, tags)
    else:
        print(f"  note: case studies need attention weights; {mode} has none")


def export_task4(cfg, device) -> None:
    from contrastive import (ContrastiveGNNBert, embed_split, retrieval_examples,
                             similarity_matrix)
    from utils import save_json

    data, loaders = _splits(cfg)
    blob = load_checkpoint(_ckpt(cfg, "task4_contrastive"), data["train"], device)

    model = ContrastiveGNNBert(data["train"].node_feature_dim, cfg).to(device)
    model.load_state_dict(blob["state_dict"])

    emb = embed_split(model, loaders["test"], device)
    sim = similarity_matrix(emb["graph"], emb["text"])
    examples = retrieval_examples(sim, emb["captions"], emb["track_ids"], n=10, top_k=3)
    save_json(examples, resolve(cfg, "retrieval") / "caption_to_audio.json")
    print(f"  wrote {len(examples)} retrieval examples -> results/retrieval_examples/")


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate export artifacts from checkpoints.")
    ap.add_argument("--task", choices=["task1", "task3", "task4"], default=None)
    ap.add_argument("--mode", default="cross_attention",
                    help="task3 fusion arm whose checkpoint to load")
    ap.add_argument("--all", action="store_true", help="every task with a checkpoint")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    if not args.task and not args.all:
        ap.error("pass --task {task1,task3,task4} or --all")

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    jobs = [args.task] if args.task else ["task1", "task3", "task4"]
    for job in jobs:
        print(f"\n== {job} ==")
        try:
            if job == "task1":
                export_task1(cfg, device)
            elif job == "task3":
                export_task3(cfg, device, args.mode)
            else:
                export_task4(cfg, device)
        except SystemExit as exc:
            if args.all:                      # a missing checkpoint is not fatal here
                print(f"  skipped: {exc}")
            else:
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
