"""Generate the two required notebooks from plain cell lists.

Keeping the notebooks in source form here means they stay diffable and can be
regenerated after an API change, instead of drifting as hand-edited JSON.

    python tools/make_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = """
import sys, json
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import load_config, load_json, get_device, resolve

CONFIG = "configs/mtat.yaml"    # <-- the preset you preprocessed with, or None for config.yaml

cfg = load_config(REPO / CONFIG if CONFIG else REPO / "config.yaml")
device = get_device(cfg.get("device", "auto"))
processed = resolve(cfg, "processed")
splits = resolve(cfg, "splits")

print("repo     :", REPO)
print("device   :", device, "| source:", cfg["dataset"]["source"],
      "| graph:", cfg["graph"]["kind"], "| segment:", cfg["audio"]["segment_seconds"], "s")
print("processed:", processed)
"""

EDA_CELLS = [
    md("""
# EDA -- music context graphs, tags and text

What this notebook covers:

1. corpus composition (genres, moods, tag frequencies)
2. what a music structure graph actually looks like
3. chord recognition from chroma
4. label sparsity -- the reason the trainers use `pos_weight` and a tuned threshold
5. confirmation that the artist-grouped split really is leak-free

Run `python src/preprocess.py --config configs/mtat.yaml` first (or whichever preset you are using -- set `CONFIG` below to match).
"""),
    code(BOOTSTRAP),
    code("""
index = load_json(processed / "index.json")
meta = load_json(processed / "meta.json")
label_space = load_json(processed / "label_space.json")

print(f"{meta['n_tracks']} tracks | {meta['num_tags']} tags | "
      f"avg {meta['avg_nodes']} nodes, {meta['avg_edges']} edges per graph")
print("splits:", meta["split_sizes"])
index[0]
"""),
    md("## 1. Corpus composition"),
    code("""
from collections import Counter

genres = Counter(r["genre"] for r in index if r["genre"])
moods = Counter(r["mood"] for r in index if r.get("mood"))
tag_counts = Counter(meta["tag_counts"])

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (title, counter) in zip(axes, [("genre", genres), ("mood", moods),
                                       ("top-15 tags", dict(tag_counts.most_common(15)))]):
    keys = list(counter)
    ax.barh(keys, [counter[k] for k in keys])
    ax.set_title(title)
    ax.invert_yaxis()
plt.tight_layout()
"""),
    md("## 2. A music structure graph\\n\\nNodes are time segments, their length set by `audio.segment_seconds`. Edges are temporal adjacency plus chroma/MFCC similarity above tau, with a size-scaled k-NN backstop. **Watch the density line**: above ~0.5 the graph is close to complete and message passing degenerates towards mean pooling."),
    code("""
from graph_builder import EDGE_TYPE_NAMES, graph_summary

track_id = index[0]["track_id"]
g = torch.load(processed / "graphs" / f"{track_id}.pt", weights_only=False)
print(json.dumps(graph_summary(g), indent=2))
print("\\ncaption:", g.text)
"""),
    code("""
# Draw it: nodes on a circle, edges coloured by type.
import matplotlib.patches as mpatches

n = int(g.num_nodes)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)
colors = {0: "#3377cc", 1: "#cc7733", 2: "#33aa66", 3: "#cccccc"}

fig, ax = plt.subplots(figsize=(6, 6))
src, dst = g.edge_index.numpy()
for s, d, t in zip(src, dst, g.edge_type.tolist()):
    if s == d:
        continue
    ax.plot(*zip(pos[s], pos[d]), color=colors.get(t, "#999"), alpha=0.6, lw=1.4)
ax.scatter(pos[:, 0], pos[:, 1], s=420, c="white", edgecolors="black", zorder=3)
for i, (x, y) in enumerate(pos):
    ax.text(x, y, str(i), ha="center", va="center", zorder=4, fontsize=9)
ax.legend(handles=[mpatches.Patch(color=c, label=EDGE_TYPE_NAMES[t])
                   for t, c in colors.items() if t != 3], loc="upper right", fontsize=8)
ax.set_title(f"segment graph -- {track_id}")
ax.set_aspect("equal"); ax.axis("off")
"""),
    md("## 3. Chord recognition from chroma\\n\\nEach segment's 12-bin chroma vector is matched against 24 triad templates."),
    code("""
from data.synthetic import synthesize_track
from audio_features import extract_track_features, chromagram
from graph_builder import recognise_chords

y, truth = synthesize_track("jazz", "calm", seed=7, duration=20.0)
tf = extract_track_features(y, "demo", cfg)
idx, names = recognise_chords(tf.chroma_segments)

print("ground-truth progression:", truth["chord_sequence"])
print("recognised from chroma :", [names[i] for i in idx])

fig, ax = plt.subplots(figsize=(11, 3))
im = ax.imshow(chromagram(y, cfg["audio"]["sample_rate"]), aspect="auto", origin="lower",
               cmap="magma")
ax.set_yticks(range(12))
ax.set_yticklabels(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
ax.set_xlabel("frame"); ax.set_title("chromagram")
plt.colorbar(im, ax=ax)
"""),
    md("## 4. Label sparsity\\n\\nMulti-label targets are mostly zeros, which is why a flat 0.5 threshold is a bad operating point."),
    code("""
Y = np.stack([np.isin(np.arange(len(label_space["tags"])),
                      [label_space["tags"].index(t) for t in r["tags"]
                       if t in label_space["tags"]]).astype(float)
              for r in index])

print(f"label density: {Y.mean():.3f}  ({Y.sum(axis=1).mean():.1f} tags per track)")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(Y.sum(axis=1), bins=range(int(Y.sum(axis=1).max()) + 2), edgecolor="k")
axes[0].set_xlabel("tags per track"); axes[0].set_title("tags per track")
axes[1].bar(range(Y.shape[1]), sorted(Y.sum(axis=0), reverse=True))
axes[1].set_xlabel("tag (sorted)"); axes[1].set_ylabel("support")
axes[1].set_title("tag support -- long tail")
plt.tight_layout()
"""),
    md("## 5. Split integrity"),
    code("""
split_ids = {name: set(load_json(splits / f"{name}.json"))
             for name in ("train", "val", "test")}
by_id = {r["track_id"]: r for r in index}
artists = {name: {by_id[t]["artist"] for t in ids if t in by_id}
           for name, ids in split_ids.items()}

print({name: len(ids) for name, ids in split_ids.items()})
print("strategy:", meta.get("split_strategy"))
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    shared = artists[a] & artists[b]
    print(f"{a} & {b}: {len(shared)} shared artists"
          + (f"  <-- LEAKAGE: {sorted(shared)[:5]}" if shared else "  OK"))
"""),
]

DEMO_CELLS = [
    md("""
# Demo -- end-to-end GNN-BERT context inference

One track, start to finish:

    waveform -> log-mel / chroma / MFCC -> segments -> graph
             +  caption -> BERT
             -> cross-attention fusion
             -> tags, valence/arousal, and the attended caption tokens

Prerequisites, for whichever corpus you trained on:

```bash
python src/preprocess.py --config configs/mtat.yaml
python src/train.py task3 --config configs/mtat.yaml
```

Set `CONFIG` in the next cell to the same preset you trained with. The
checkpoint records the corpus it was trained on, so a mismatch between the
weights and the processed data is reported rather than crashing on a tensor
shape.
"""),
    code("""
import sys, json
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import load_config, load_json, get_device, resolve

CONFIG = "configs/mtat.yaml"      # <-- the preset you trained with, or None for config.yaml
RESULTS = None                    # <-- override the results dir, or None to take it from the config

cfg = load_config(REPO / CONFIG if CONFIG else REPO / "config.yaml")
device = get_device(cfg.get("device", "auto"))

processed = resolve(cfg, "processed")
splits = resolve(cfg, "splits")
results = Path(RESULTS) if RESULTS else resolve(cfg, "results")

print("repo     :", REPO)
print("device   :", device)
print("corpus   :", cfg["dataset"]["source"], "| graph:", cfg["graph"]["kind"],
      "| segment:", cfg["audio"]["segment_seconds"], "s")
print("processed:", processed)
print("results  :", results)
"""),
    md("""
## 1. Take an audio clip

By default this picks a real clip from the **test** split, so the demo shows
inference on data the model never saw. Set `AUDIO_PATH` to run on any file of
your own, or `USE_SYNTHETIC = True` to compose one.
"""),
    code("""
from audio_features import load_audio, extract_track_features
from data.synthetic import synthesize_track, build_caption, build_tags

AUDIO_PATH = None        # e.g. "../data/raw/gtzan/genres_original/jazz/jazz.00000.wav"
USE_SYNTHETIC = False

index = {r["track_id"]: r for r in load_json(processed / "index.json")}
test_ids = load_json(splits / "test.json")

record, truth = None, {}
if AUDIO_PATH:
    y = load_audio(AUDIO_PATH, cfg["audio"]["sample_rate"], cfg["dataset"]["clip_seconds"])
    caption = "a music track"
elif USE_SYNTHETIC:
    y, truth = synthesize_track("jazz", "melancholic", seed=1234,
                                duration=cfg["dataset"]["clip_seconds"])
    caption = build_caption(truth, np.random.default_rng(0), label_dropout=1.0)
else:
    # First test clip whose source audio is still on disk.
    record = next((index[t] for t in test_ids
                   if index.get(t, {}).get("audio_path")
                   and Path(index[t]["audio_path"]).exists()), None)
    if record is None:
        raise SystemExit("no test clip has its audio on disk -- set AUDIO_PATH "
                         "or USE_SYNTHETIC = True")
    y = load_audio(record["audio_path"], cfg["audio"]["sample_rate"],
                   cfg["dataset"]["clip_seconds"])
    caption = record["text"]

print("samples :", y.shape[0])
print("caption :", caption)
if record:
    print("track   :", record["track_id"], "| artist:", record["artist"])
    print("true tags:", record["tags"])
elif truth:
    print("true tags:", build_tags(truth))
"""),
    md("## 2. Features and graph"),
    code("""
from graph_builder import build_graph, graph_summary

tf = extract_track_features(y, "demo_track", cfg)
graph = build_graph(tf, cfg)
print(json.dumps(graph_summary(graph), indent=2))
print("segment times:", tf.segment_times())
"""),
    md("## 3. Tokenise the caption and assemble one batch"),
    code("""
from transformers import AutoTokenizer
from torch_geometric.data import Batch

tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model_name"])
enc = tokenizer([caption], padding="max_length", truncation=True,
                max_length=cfg["text"]["max_length"], return_tensors="pt")

tags = load_json(processed / "label_space.json")["tags"]

graph.input_ids = enc["input_ids"]
graph.attention_mask = enc["attention_mask"]
graph.y = torch.zeros(1, len(tags))
graph.va = torch.zeros(1, 2)
graph.va_mask = torch.zeros(1)
graph.text = caption

batch = Batch.from_data_list([graph]).to(device)
batch
"""),
    md("""
## 4. Load the trained fusion model

`load_checkpoint` compares the checkpoint's recorded corpus and label-space
size against the processed data and raises `CheckpointMismatch` with an
explanation if they disagree — rather than failing inside `load_state_dict`
with a bare tensor-shape error.
"""),
    code("""
from fusion_model import build_fusion_model, top_attended_tokens
from engine import emotion_stats, load_checkpoint
from data.dataset import MusicContextDataset

ckpt_path = results / "checkpoints" / "task3_fusion_cross_attention.pt"
assert ckpt_path.exists(), (
    f"no checkpoint at {ckpt_path}\\n"
    f"train it first:  python src/train.py task3 --config {CONFIG}")

train_ds = MusicContextDataset(processed, splits, "train")
predict_emotion = bool(train_ds.meta.get("has_emotion"))

blob = load_checkpoint(ckpt_path, train_ds, device)     # guards the label space
print("checkpoint provenance:", json.dumps(blob.get("provenance"), indent=2))

model = build_fusion_model(cfg, train_ds.node_feature_dim,
                           len(tags), predict_emotion).to(device)
model.load_state_dict(blob["state_dict"])
model.eval()

with torch.no_grad():
    out = model(batch)
    probs = torch.sigmoid(out["tag_logits"])[0].cpu().numpy()
print("\\nloaded:", ckpt_path.name, "| best epoch:", blob.get("best_epoch"))
"""),
    md("## 5. Predictions\\n\\nThe threshold is the one tuned on the validation split during training -- not a flat 0.5, which is a poor operating point for sparse multi-label targets."),
    code("""
metrics = load_json(results / "metrics.json")
threshold = metrics["task3_fusion_cross_attention"]["threshold"]
order = np.argsort(-probs)[:10]

print(f"threshold tuned on val = {threshold:.2f}\\n")
print(f"{'tag':<18}{'score':>8}   predicted")
for i in order:
    print(f"{tags[i]:<18}{probs[i]:>8.4f}   {'YES' if probs[i] >= threshold else ''}")

if predict_emotion and out["emotion"] is not None:
    mean, std = emotion_stats(train_ds)
    va = out["emotion"][0].cpu() * std + mean
    print(f"\\nvalence {va[0]:.2f} / arousal {va[1]:.2f}  (DEAM 1-9 scale)")
"""),
    md("## 6. What the model attended to\\n\\nThe graph vector queries the caption tokens, so the attention weights say which words the audio structure aligned with."),
    code("""
attended = top_attended_tokens(tokenizer, batch.input_ids, out["attention"], k=8)
if attended:
    for token, weight in attended[0]:
        bar = "#" * int(weight * 200)
        print(f"{token:<14}{weight:.4f}  {bar}")
else:
    print("no attention (fusion.mode is not cross_attention)")
"""),
    code("""
fig, ax = plt.subplots(figsize=(10, 3))
top = order[:10]
ax.barh([tags[i] for i in top], probs[top])
ax.axvline(threshold, color="crimson", ls="--", label=f"threshold {threshold:.2f}")
ax.invert_yaxis(); ax.set_xlabel("probability"); ax.legend()
ax.set_title("predicted context tags")
plt.tight_layout()
"""),
]


COLAB_CELLS = [
    md("""
# GPU run — MagnaTagATune + GTZAN

Runs the whole experimental suite on a free Colab T4. CPU-only training of the
BERT-based tasks takes 6–8 hours; this takes roughly 40–70 minutes.

**Before you start:** Runtime → Change runtime type → Hardware accelerator → **T4 GPU**.

What this does:

1. clones the repo and installs the pinned dependencies
2. downloads MagnaTagATune (~2.8 GB) *inside* Colab — far faster than uploading
3. Tasks 1, 3 (4-arm ablation) and 4 + baselines on **MagnaTagATune** (the spec's
   corpus for Tasks 1 and 3)
4. Task 2 + GAT + a segment/chord/hybrid graph comparison on **GTZAN** (the
   spec's corpus for Task 2)
5. zips `results_*/` for download

Each corpus writes to its own results directory — mixing corpora in one metrics
file is a hard error, by design.
"""),
    code("""
!nvidia-smi -L
import torch, os
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then re-run."
"""),
    md("""
## 1. Get the code

The repo URL is already filled in. **If the repo is private**, paste a GitHub
personal access token into `TOKEN` in the next cell — Colab cannot clone a
private repo without one. Leave `TOKEN` empty if the repo is public.

(Token: GitHub → Settings → Developer settings → Personal access tokens →
Tokens (classic) → Generate new token → tick `repo`.)
"""),
    code("""
REPO = "sharminahmednova/Supervised-Neural-Network-Project-GNN-Based-BERT-for-Understanding-Context-from-Music"

# Only needed if the repo is PRIVATE: GitHub > Settings > Developer settings >
# Personal access tokens > Tokens (classic), tick `repo`. Leave "" if public.
TOKEN = ""

REPO_URL = (f"https://{TOKEN}@github.com/{REPO}.git" if TOKEN
            else f"https://github.com/{REPO}.git")

# Clone into a FIXED directory. `git clone` otherwise names the folder after
# the repo, so every path below would depend on how long the repo name is.
WORKDIR = "/content/project"

import os, subprocess
from pathlib import Path

if (Path(WORKDIR) / ".git").exists():
    subprocess.run(["git", "-C", WORKDIR, "pull", "-q"], check=False)
    print("updated existing clone")
else:
    subprocess.run(["git", "clone", "-q", REPO_URL, WORKDIR], check=True)
    print("cloned")

os.chdir(WORKDIR)
print("cwd:", Path.cwd())
subprocess.run(["ls"])
"""),
    code("""
# ALTERNATIVE to the clone above, if the clone fails (private repo, no token).
# Zip your local project folder, upload it here, and carry on.
# import os, glob, subprocess
# from google.colab import files
# from pathlib import Path
# files.upload()                                  # pick your .zip
# zip_name = sorted(glob.glob("*.zip"))[0]
# subprocess.run(["unzip", "-q", "-o", zip_name, "-d", "/content/unzipped"])
# # the zip may or may not contain a top-level folder -- find the real root
# root = next(p.parent for p in Path("/content/unzipped").rglob("config.yaml"))
# os.chdir(root); print("cwd:", Path.cwd())
"""),
    md("## 2. Dependencies"),
    code("""
# Colab already ships a CUDA torch; install the rest pinned.
!pip install -q torch-geometric==2.8.0.post1 transformers==5.17.0 librosa==0.11.0 soundfile==0.14.0 pyyaml==6.0.3
!python tools/check_env.py
"""),
    md("## 3. Fetch the corpora\n\nMagnaTagATune is ~2.8 GB as a 3-part split zip; GTZAN is ~1.1 GB. Colab's link makes this minutes rather than hours."),
    code("""
!python tools/fetch_data.py magnatagatune --extract
!python tools/fetch_data.py gtzan --extract
!du -sh data/raw/*
"""),
    md("""
## 4. MagnaTagATune — Tasks 1, 3, 4 + baselines

`configs/mtat.yaml` sets the honest configuration:

- `top_tags: 50` — the spec's top-50 subset
- `text_tag_fraction: 0.4` — MTAT has no captions, so 40% of the vocabulary
  becomes the "caption" and the **disjoint** remainder is the prediction target.
  The text therefore cannot leak the labels it is scored against. At 0.0 the
  text restates the target and Task 1 scores a meaningless 1.0.
- `segment_seconds: 2.5` — 29 s clips at 5 s windows give a *six node* graph at
  ~0.54 density, where message passing ≈ mean pooling. 2.5 s gives 12 nodes at
  ~0.27 density.
"""),
    code("""
!python src/preprocess.py --config configs/mtat.yaml --export-samples 20
!python -c "import json;m=json.load(open('data/processed_mtat/meta.json'));print({k:v for k,v in m.items() if k!='tag_counts'})"
"""),
    code("""
# GPU: larger batch and a full schedule are affordable here.
!python src/train.py all --config configs/mtat.yaml \\
    --set device=cuda --set train.epochs=30 --set train.batch_size=64 \\
    --set train.patience=8 --set text.unfreeze_last_n=4
"""),
    md("""
### Task 4 on MagnaTagATune -- needs its own preprocessing

~35% of MTAT records have no text-side tag after the disjoint partition and
share one placeholder string. Identical text ties exactly in retrieval (~61
ties per query, measured), so retrieval must be run on a corpus with those
records dropped. This writes to its own processed/results directories.
"""),
    code("""
!python src/preprocess.py --config configs/mtat.yaml \
    --set dataset.drop_textless=true \
    --set paths.processed=data/processed_mtat_r4 \
    --set paths.splits=data/splits_mtat_r4 --export-samples 0
!python src/train.py task4 --config configs/mtat.yaml \
    --set dataset.drop_textless=true \
    --set paths.processed=data/processed_mtat_r4 \
    --set paths.splits=data/splits_mtat_r4 \
    --set paths.results=results_mtat_r4 \
    --set device=cuda --set train.epochs=40 --set train.batch_size=64 \
    --set train.patience=10 --set text.unfreeze_last_n=4
"""),
    md("""
### Does cross-attention actually beat early concat?

On the synthetic corpus early concat (0.837) beat cross-attention (0.807), and
cross-attention did not beat `bert_only` (0.809) — but its validation curve was
still rising at the epoch cap, so undertraining is a live alternative
explanation. This cell gives the cross-attention arm twice the schedule; if it
still does not win, the negative result is real and worth reporting.
"""),
    code("""
!python src/train.py task3 --config configs/mtat.yaml --mode cross_attention \\
    --set device=cuda --set train.epochs=60 --set train.batch_size=64 \\
    --set train.patience=12 --set text.unfreeze_last_n=4 \\
    --set paths.results=results_mtat_longsched
"""),
    md("## 5. GTZAN — Task 2 genre classification\n\nThe spec asks for genre classification on GTZAN or FMA-small. Tasks 1/3/4 are *not* run here: GTZAN's only text is its genre label, so the text branch trivially reproduces the target."),
    code("""
!python src/preprocess.py --config configs/gtzan.yaml --export-samples 20
!python src/train.py baselines --config configs/gtzan.yaml --set device=cuda --set train.epochs=30 --set train.batch_size=64
!python src/train.py task2 --config configs/gtzan.yaml --set device=cuda --set train.epochs=30 --set train.batch_size=64
!python src/train.py task2 --config configs/gtzan.yaml --set device=cuda --set gnn.conv=gat --set train.epochs=30 --set train.batch_size=64
"""),
    md("### Graph construction comparison\n\nThe spec calls out chord-transition graphs specifically. Each `graph.kind` needs its own preprocessing, so each writes to its own directory."),
    code("""
# subprocess rather than `!` -- a shell escape inside a Python loop with line
# continuations is the kind of thing that silently runs the wrong command.
import subprocess, sys

for kind in ["segment", "chord", "hybrid"]:
    for cmd in (
        ["python", "src/preprocess.py", "--config", "configs/gtzan.yaml",
         "--set", f"graph.kind={kind}",
         "--set", f"paths.processed=data/processed_gtzan_{kind}",
         "--set", f"paths.splits=data/splits_gtzan_{kind}",
         "--export-samples", "0"],
        ["python", "src/train.py", "task2", "--config", "configs/gtzan.yaml",
         "--set", "device=cuda", "--set", f"graph.kind={kind}",
         "--set", f"paths.processed=data/processed_gtzan_{kind}",
         "--set", f"paths.splits=data/splits_gtzan_{kind}",
         "--set", f"paths.results=results_gtzan_{kind}",
         "--set", "train.epochs=30", "--set", "train.batch_size=64"],
    ):
        print(">", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode:
            print(f"  FAILED ({kind}) -- continuing with the other graph kinds")
            break
"""),
    md("""
## 6. DEAM — the auxiliary emotion term

The spec's multi-task loss is
`L = L_tags + α‖v − v̂‖² + β‖a − â‖²`, with valence/arousal from DEAM. DEAM
carries no tags, so read the MAE/R² here, not the F1.
"""),
    code("""
!python tools/fetch_data.py deam --extract
!python src/preprocess.py --config configs/deam.yaml --export-samples 20
!python src/train.py task3 --config configs/deam.yaml --mode cross_attention \\
    --set device=cuda --set train.epochs=30 --set train.batch_size=64
"""),
    md("""
## 7. MusicCaps — Task 4's own corpus

MusicCaps ships captions and YouTube ids, not audio (it is not
redistributable), so clips are pulled individually with `yt-dlp`. Colab already
has `ffmpeg`, which the local Windows machine does not — this is the right place
to try it.

Expect losses: some videos are gone, and YouTube rate-limits. The fetcher keeps
a resume manifest, reports its success rate, and stops early if the rate
collapses. **If fewer than ~200 clips land, run Task 4 on MagnaTagATune instead
and say so in the report** — a documented deviation costs far less than a table
built on 40 clips.
"""),
    code("""
!pip install -q yt-dlp
!python tools/fetch_data.py musiccaps --clips 1200
"""),
    code("""
import glob, subprocess

n = len(glob.glob('data/raw/musiccaps/audio/*.wav'))
print(f"{n} MusicCaps clips on disk")

# MusicCaps clips are only 10 s, so segments must be ~1 s to give a graph at
# all -- at 2.5 s a clip is four nodes, which is not a graph worth message
# passing over.
COMMON = ["--set", "dataset.source=musiccaps",
          "--set", "paths.processed=data/processed_musiccaps",
          "--set", "paths.splits=data/splits_musiccaps",
          "--set", "audio.segment_seconds=1.0"]

if n >= 200:
    for cmd in (
        ["python", "src/preprocess.py", *COMMON,
         "--set", "dataset.clip_seconds=10",
         "--set", "dataset.min_tag_count=20",
         "--set", "dataset.text_field=caption",
         "--set", "dataset.text_tag_fraction=0.0",
         "--export-samples", "20"],
        ["python", "src/train.py", "task4", *COMMON,
         "--set", "paths.results=results_musiccaps",
         "--set", "device=cuda", "--set", "train.epochs=40",
         "--set", "train.batch_size=64"],
    ):
        print(">", " ".join(cmd))
        if subprocess.run(cmd).returncode:
            print("  FAILED -- fall back to the MagnaTagATune Task 4 result")
            break
else:
    print("\\nToo few clips for a meaningful retrieval split. Use the "
          "MagnaTagATune Task 4\\nresult and state the deviation in the "
          "report's Limitations (there is a \\\\FILL\\nmarker for exactly this).")
"""),
    md("""
## 8. Build the Task 4 listening study

Exports the retrieved clips and writes one self-contained HTML file. Send it to
at least 5 listeners (the spec's minimum), collect the `ratings_*.json` files
they download, and aggregate.
"""),
    code("""
# Prefer MusicCaps (real captions); else the textless-dropped MTAT retrieval run.
import os

CANDIDATES = [
    ("results_musiccaps", "data/processed_musiccaps"),
    ("results_mtat_r4",   "data/processed_mtat_r4"),
    ("results_mtat",      "data/processed_mtat"),
]
RES, PROC = next(((r, p) for r, p in CANDIDATES
                  if os.path.exists(f"{r}/retrieval_examples")), (None, None))
assert RES, "no Task 4 retrieval output found -- run a task4 cell first"
print("building study from", RES)

!python tools/make_human_eval.py --results $RES --processed $PROC --queries 10 --seconds 12

from google.colab import files
files.download(f"{RES}/human_eval/rating_form.html")
"""),
    md("## 9. Tables and plots"),
    code("""
!python src/evaluate.py --set paths.results=results_mtat --set paths.plots=results_mtat/plots
!python src/evaluate.py --set paths.results=results_gtzan --set paths.plots=results_gtzan/plots
"""),
    md("## 10. Download the results\n\nOnly `results_*` comes back — the graphs and raw audio stay here."),
    code("""
!zip -qr results_gpu.zip results_mtat results_mtat_r4 results_gtzan results_gtzan_* \
    results_mtat_longsched results_deam results_musiccaps \\
    data/processed_mtat/samples data/processed_gtzan/samples \\
    data/processed_mtat/meta.json data/processed_mtat/label_space.json \\
    data/processed_gtzan/meta.json data/processed_gtzan/label_space.json 2>/dev/null
!du -sh results_gpu.zip

from google.colab import files
files.download("results_gpu.zip")
"""),
    md("""
## Back on your machine

```bash
unzip -o results_gpu.zip -d .
python src/evaluate.py --set paths.results=results_mtat --set paths.plots=results_mtat/plots
python tools/make_report_tables.py --results results_mtat
```

Then `report/main.tex` picks the numbers up automatically.
"""),
]


KAGGLE_CELLS = [
    md("""
# GPU run on Kaggle — MagnaTagATune + GTZAN

Same experiments as `colab_run.ipynb`, trimmed to **only what carries marks**
so it finishes in about 2 hours instead of most of a day.

**Before you start — three settings in the right-hand panel:**

1. **Accelerator → GPU T4 x2** (or P100)
2. **Internet → On** &nbsp;← *without this, nothing downloads and every cell fails*
3. **Persistence → Files only** (optional, lets you resume)

Kaggle gives 30 GPU-hours a week and 9 hours per session, so this fits easily.

## What runs

| | Corpus | Why |
|---|---|---|
| Baselines B1/B2/B4 | MagnaTagATune | required comparison |
| Task 1 — BERT tagging | MagnaTagATune | spec's corpus for Task 1 |
| Task 3 — fusion, 4-arm ablation | MagnaTagATune | spec's corpus for Task 3 |
| Task 4 — contrastive retrieval | MagnaTagATune | textless clips dropped |
| Task 2 — genre classification | GTZAN | spec's corpus for Task 2 |

Optional extras (GAT, chord-graph comparison, DEAM, MusicCaps) are at the end,
switched off by default. Turn them on only if you have time to spare.
"""),
    code("""
import subprocess, sys
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "No GPU. Right-hand panel > Accelerator > GPU T4 x2, then re-run.")
"""),
    md("## 1. Get the code\n\nKaggle's writable directory is `/kaggle/working`. Everything below lives there."),
    code("""
REPO = "sharminahmednova/Supervised-Neural-Network-Project-GNN-Based-BERT-for-Understanding-Context-from-Music"
TOKEN = ""    # only needed if the repo is PRIVATE

REPO_URL = (f"https://{TOKEN}@github.com/{REPO}.git" if TOKEN
            else f"https://github.com/{REPO}.git")
WORKDIR = "/kaggle/working/project"

import os, subprocess
from pathlib import Path

if (Path(WORKDIR) / ".git").exists():
    subprocess.run(["git", "-C", WORKDIR, "pull", "-q"], check=False)
    print("updated existing clone")
else:
    subprocess.run(["git", "clone", "-q", REPO_URL, WORKDIR], check=True)
    print("cloned")

os.chdir(WORKDIR)
print("cwd:", Path.cwd())
subprocess.run(["ls"])
"""),
    md("""
If the clone fails with a DNS or connection error, **Internet is off**. Turn it
on in the right-hand panel (it requires a phone-verified Kaggle account) and
re-run this cell.
"""),
    md("## 2. Dependencies\n\nKaggle already ships a CUDA torch; only the rest is installed."),
    code("""
!pip install -q torch-geometric==2.8.0.post1 transformers==5.17.0 librosa==0.11.0 soundfile==0.14.0 pyyaml==6.0.3
!python tools/check_env.py
"""),
    md("""
## 3. Fetch the corpora

MagnaTagATune is ~2.8 GB and GTZAN ~1.1 GB. Kaggle's `/kaggle/working` holds
20 GB, which is enough for both plus the derived graphs.
"""),
    code("""
!python tools/fetch_data.py magnatagatune --extract
!python tools/fetch_data.py gtzan --extract
!du -sh data/raw/*
"""),
    md("""
## 4. MagnaTagATune — baselines, Task 1, Task 3

`SUBSAMPLE` is the main time dial. 4,000 clips takes roughly 90 minutes end to
end and is plenty for the report; raise it to 8,000 if you have hours to spare.
"""),
    code("""
SUBSAMPLE = 4000
EPOCHS    = 20
BATCH     = 64

COMMON = ["--config", "configs/mtat.yaml", "--set", "device=cuda",
          "--set", f"train.epochs={EPOCHS}", "--set", f"train.batch_size={BATCH}",
          "--set", "train.patience=6", "--set", "text.unfreeze_last_n=4"]

import subprocess, time

def run(cmd, label):
    print(f"\\n{'='*70}\\n{label}\\n{'='*70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print(f"[{label}] {'OK' if r.returncode == 0 else 'FAILED'} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    return r.returncode == 0
"""),
    code("""
run(["python", "src/preprocess.py", "--config", "configs/mtat.yaml",
     "--set", f"dataset.subsample={SUBSAMPLE}", "--export-samples", "20"],
    "preprocess MagnaTagATune")
"""),
    code("""
run(["python", "src/train.py", "baselines", *COMMON], "baselines (B1, B2, B4)")
"""),
    code("""
run(["python", "src/train.py", "task1", *COMMON], "Task 1 - BERT tagging")
"""),
    code("""
# The 4-arm ablation: bert_only / gnn_only / concat / cross_attention.
# This is the longest cell -- roughly 45 min at SUBSAMPLE=4000.
run(["python", "src/train.py", "task3", "--ablation", *COMMON],
    "Task 3 - fusion ablation")
"""),
    md("""
### Task 4 — retrieval, on its own preprocessing

~35% of MagnaTagATune clips have no text-side tag after the disjoint tag split
and share one placeholder string. Identical text ties exactly in retrieval, so
those clips are dropped here. Writes to `results_mtat_r4`.
"""),
    code("""
R4 = ["--set", "dataset.drop_textless=true",
      "--set", "paths.processed=data/processed_mtat_r4",
      "--set", "paths.splits=data/splits_mtat_r4"]

if run(["python", "src/preprocess.py", "--config", "configs/mtat.yaml",
        "--set", f"dataset.subsample={SUBSAMPLE}", *R4, "--export-samples", "0"],
       "preprocess MTAT for retrieval"):
    run(["python", "src/train.py", "task4", *COMMON, *R4,
         "--set", "paths.results=results_mtat_r4"],
        "Task 4 - contrastive retrieval")
"""),
    md("## 5. GTZAN — Task 2 genre classification\n\nFast: 1,000 clips, no BERT in the loop."),
    code("""
GTZAN = ["--config", "configs/gtzan.yaml", "--set", "device=cuda",
         "--set", f"train.epochs={EPOCHS}", "--set", f"train.batch_size={BATCH}"]

run(["python", "src/preprocess.py", "--config", "configs/gtzan.yaml",
     "--export-samples", "20"], "preprocess GTZAN")
run(["python", "src/train.py", "baselines", *GTZAN], "GTZAN baselines")
run(["python", "src/train.py", "task2", *GTZAN], "Task 2 - GNN genre classification")
"""),
    md("## 6. Tables and plots"),
    code("""
!python src/evaluate.py --set paths.results=results_mtat --set paths.plots=results_mtat/plots
!python src/evaluate.py --set paths.results=results_gtzan --set paths.plots=results_gtzan/plots
!python tools/make_report_tables.py --results results_mtat
!cat results_mtat/comparison_table.txt
"""),
    md("""
## 7. Package the results

Kaggle has no `files.download()`. Instead, write the zip to `/kaggle/working`
and grab it from the **Output** tab on the right (or *Save Version → Output*).
"""),
    code("""
import shutil, subprocess
from pathlib import Path

OUT = Path("/kaggle/working/results_gpu")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

for name in ["results_mtat", "results_mtat_r4", "results_gtzan"]:
    src = Path(name)
    if src.exists():
        shutil.copytree(src, OUT / name, dirs_exist_ok=True)

# The 20 sample graphs and the corpus metadata are submission items.
for proc in ["data/processed_mtat", "data/processed_gtzan"]:
    p = Path(proc)
    if not p.exists():
        continue
    dst = OUT / proc
    dst.mkdir(parents=True, exist_ok=True)
    if (p / "samples").exists():
        shutil.copytree(p / "samples", dst / "samples", dirs_exist_ok=True)
    for meta in ["meta.json", "label_space.json", "index.json"]:
        if (p / meta).exists():
            shutil.copy2(p / meta, dst / meta)

shutil.make_archive("/kaggle/working/results_gpu", "zip", OUT)
shutil.rmtree(OUT)
size = Path("/kaggle/working/results_gpu.zip").stat().st_size / 1e6
print(f"results_gpu.zip  ({size:.1f} MB)")
print("Download it from the Output panel on the right.")
"""),
    md("""
## 8. Optional extras

Everything above is what the marks depend on. These add breadth if you have
session time left — flip a flag to `True` and re-run the cell.
"""),
    code("""
RUN_GAT          = False   # GAT instead of GraphSAGE on Task 2
RUN_GRAPH_KINDS  = False   # segment vs chord vs hybrid graphs
RUN_DEAM         = False   # valence/arousal auxiliary loss (~25 min, +1.7 GB)
RUN_MUSICCAPS    = False   # Task 4 on real captions (slow, YouTube often blocks)

if RUN_GAT:
    run(["python", "src/train.py", "task2", *GTZAN, "--set", "gnn.conv=gat",
         "--set", "paths.results=results_gtzan_gat"], "Task 2 with GAT")

if RUN_GRAPH_KINDS:
    for kind in ["segment", "chord", "hybrid"]:
        ok = run(["python", "src/preprocess.py", "--config", "configs/gtzan.yaml",
                  "--set", f"graph.kind={kind}",
                  "--set", f"paths.processed=data/processed_gtzan_{kind}",
                  "--set", f"paths.splits=data/splits_gtzan_{kind}",
                  "--export-samples", "0"], f"preprocess GTZAN ({kind})")
        if ok:
            run(["python", "src/train.py", "task2", *GTZAN,
                 "--set", f"graph.kind={kind}",
                 "--set", f"paths.processed=data/processed_gtzan_{kind}",
                 "--set", f"paths.splits=data/splits_gtzan_{kind}",
                 "--set", f"paths.results=results_gtzan_{kind}"],
                f"Task 2 ({kind} graph)")

if RUN_DEAM:
    if run(["python", "tools/fetch_data.py", "deam", "--extract"], "fetch DEAM"):
        run(["python", "src/preprocess.py", "--config", "configs/deam.yaml",
             "--export-samples", "20"], "preprocess DEAM")
        run(["python", "src/train.py", "task3", "--config", "configs/deam.yaml",
             "--mode", "cross_attention", "--set", "device=cuda",
             "--set", f"train.epochs={EPOCHS}", "--set", f"train.batch_size={BATCH}"],
            "Task 3 with DEAM emotion target")

if RUN_MUSICCAPS:
    !pip install -q yt-dlp
    run(["python", "tools/fetch_data.py", "musiccaps", "--clips", "800"],
        "fetch MusicCaps")
"""),
    md("""
## 9. The listening study

Build it here so the clips come from the same run as your numbers, then
download `rating_form.html` from the Output panel and send it to 5 people.
"""),
    code("""
import os
from pathlib import Path

RES, PROC = next(((r, p) for r, p in [
    ("results_musiccaps", "data/processed_musiccaps"),
    ("results_mtat_r4",   "data/processed_mtat_r4"),
    ("results_mtat",      "data/processed_mtat"),
] if Path(r, "retrieval_examples").exists()), (None, None))

assert RES, "no Task 4 retrieval output -- run the Task 4 cell first"
print("building the study from", RES)

!python tools/make_human_eval.py --results $RES --processed $PROC --queries 10 --seconds 12
!cp $RES/human_eval/rating_form.html /kaggle/working/rating_form.html
print("\\nrating_form.html is in the Output panel -- send that one file to 5 listeners.")
"""),
    md("""
## Back on your machine

Download **`results_gpu.zip`** and **`rating_form.html`** from the Output panel,
put the zip in your project folder, then:

```bash
unzip -o results_gpu.zip -d .
python src/evaluate.py --set paths.results=results_mtat --set paths.plots=results_mtat/plots
python tools/make_report_tables.py --results results_mtat
```
"""),
]


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in (("eda.ipynb", EDA_CELLS), ("demo_context.ipynb", DEMO_CELLS),
                        ("colab_run.ipynb", COLAB_CELLS),
                        ("kaggle_run.ipynb", KAGGLE_CELLS)):
        path = NB_DIR / name
        path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
