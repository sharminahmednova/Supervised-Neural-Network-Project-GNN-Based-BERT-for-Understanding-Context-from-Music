"""CLI: build graphs, tokenise text, and write splits.

    python src/preprocess.py                          # uses config.yaml
    python src/preprocess.py --set dataset.source=gtzan
    python src/preprocess.py --limit 20 --export-samples 20
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import preprocess  # noqa: E402
from utils import REPO_ROOT, load_config, resolve, save_json, set_seed  # noqa: E402


def export_samples(processed_root: Path, n: int) -> Path:
    """Copy N example graphs to data/processed/samples/ (submission item 2)."""
    import torch

    from graph_builder import graph_summary

    src = processed_root / "graphs"
    dst = processed_root / "samples"
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("*"):
        f.unlink()

    picked = sorted(src.glob("*.pt"))[:n]
    summaries = []
    for f in picked:
        shutil.copy2(f, dst / f.name)
        g = torch.load(f, weights_only=False)
        s = graph_summary(g)
        s["tags"] = [t for t, v in zip(range(g.y.numel()), g.y.reshape(-1).tolist()) if v]
        s["text"] = getattr(g, "text", "")
        summaries.append(s)
    save_json(summaries, dst / "summary.json")
    print(f"  exported {len(picked)} sample graphs -> {dst}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess audio + text into graphs.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="override any config key, repeatable")
    ap.add_argument("--limit", type=int, default=None, help="cap tracks (smoke tests)")
    ap.add_argument("--export-samples", type=int, default=20,
                    help="how many example graphs to copy out (0 disables)")
    ap.add_argument("--clean", action="store_true", help="wipe processed/ first")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.get("seed", 42))

    raw = resolve(cfg, "raw")
    processed = resolve(cfg, "processed")
    splits = resolve(cfg, "splits")

    if args.clean and (processed / "graphs").exists():
        shutil.rmtree(processed / "graphs")

    print(f"source={cfg['dataset']['source']}  graph={cfg['graph']['kind']}  "
          f"text={cfg['text']['model_name']}")
    meta = preprocess(cfg, raw, processed, splits, limit=args.limit)

    if args.export_samples:
        export_samples(processed, args.export_samples)

    save_json(meta, REPO_ROOT / "results" / "preprocess_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
