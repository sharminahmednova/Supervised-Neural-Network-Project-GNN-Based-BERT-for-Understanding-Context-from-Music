"""Shared helpers: config loading, seeding, device selection, IO."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """dict with attribute access and dotted-path get/set."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _coerce(text: str) -> Any:
    """Turn a CLI string into bool/int/float/None where it clearly is one."""
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively overlay `overlay` onto `base`, returning a new dict."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_chain(path: Path, _seen: set | None = None) -> dict:
    """Load a YAML config, resolving an ``extends:`` chain first.

    Per-corpus presets in `configs/` are overlays, not copies -- a copy of the
    whole config drifts out of sync with `config.yaml` the moment either moves.
    """
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular extends chain at {path}")
    _seen.add(path)

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path)
    return _deep_merge(_load_yaml_chain(parent_path, _seen), raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load config.yaml (or a preset that ``extends`` it), then apply overrides."""
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    cfg = Config(_load_yaml_chain(path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        cfg.set_path(key.strip(), _coerce(raw.strip()))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        pref = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(pref)


def resolve(cfg: Config, key: str) -> Path:
    """Resolve a `paths.*` entry to an absolute path, creating it if needed."""
    p = REPO_ROOT / cfg.get_path(f"paths.{key}", key)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def build_provenance(cfg: "Config", meta: dict | None = None) -> dict:
    """Identify the corpus and preprocessing a metric row was produced on.

    Every metrics row carries one of these. Without it, rows from different
    corpora can silently end up in one table -- which happened here: GTZAN
    baselines (150 test clips, 10 tags) were merged with synthetic task rows
    (90 clips, 32 tags), producing a file that read as though GNN-BERT had
    beaten a CNN on GTZAN when that was never measured.
    """
    meta = meta or {}
    return {
        "corpus": meta.get("source", cfg.get_path("dataset.source", "?")),
        "n_tracks": meta.get("n_tracks"),
        "num_tags": meta.get("num_tags"),
        "graph_kind": meta.get("graph_kind", cfg.get_path("graph.kind")),
        "segment_seconds": cfg.get_path("audio.segment_seconds"),
        "split_sizes": meta.get("split_sizes"),
        "split_strategy": meta.get("split_strategy"),
        "text_model": meta.get("text_model", cfg.get_path("text.model_name")),
        "seed": cfg.get("seed"),
    }


def _provenance_key(prov: dict) -> tuple:
    """The fields that must agree for two rows to belong in the same table."""
    return (prov.get("corpus"), prov.get("num_tags"), prov.get("graph_kind"),
            prov.get("segment_seconds"))


class ProvenanceConflict(RuntimeError):
    """Raised when a metrics file would end up describing two different corpora."""


def merge_metrics(path: str | Path, key: str, payload: dict,
                  provenance: dict | None = None) -> dict:
    """Accumulate per-run metrics into a single results/metrics.json.

    `provenance` is required: it is stamped onto the row and checked against the
    file's existing rows, so a metrics file can only ever describe one corpus.
    Point `paths.results` elsewhere (or use a corpus-suffixed filename) to score
    a second corpus.
    """
    if provenance is None:
        raise ValueError(
            "merge_metrics() requires provenance -- build it with "
            "utils.build_provenance(cfg, dataset.meta). See ProvenanceConflict.")

    path = Path(path)
    blob = load_json(path) if path.exists() else {}

    existing = blob.get("_provenance")
    if existing and _provenance_key(existing) != _provenance_key(provenance):
        raise ProvenanceConflict(
            f"{path.name} already holds results for "
            f"{existing.get('corpus')} ({existing.get('num_tags')} tags, "
            f"graph={existing.get('graph_kind')}, "
            f"seg={existing.get('segment_seconds')}s) but this run is "
            f"{provenance.get('corpus')} ({provenance.get('num_tags')} tags, "
            f"graph={provenance.get('graph_kind')}, "
            f"seg={provenance.get('segment_seconds')}s).\n"
            "Mixing corpora in one metrics file produces a table that cannot be "
            "interpreted. Write this run elsewhere, e.g.\n"
            f"  --set paths.results=results_{provenance.get('corpus')}")

    blob["_provenance"] = provenance
    blob[key] = {**payload, "_provenance": provenance}
    save_json(blob, path)
    return blob


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
