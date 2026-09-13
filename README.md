# GNN-BERT for Understanding Context from Music

Supervised neural network project for **Neural Networks and Fuzzy Systems**
(CSE425 / EEE474 / CSE715).

A hybrid **BERT + Graph Neural Network** system that predicts musical context —
genre, mood, instrumentation, and continuous valence/arousal — by combining
*semantic* text representations with *relational* audio structure.

| Task | Model | Corpus | Code |
|---|---|---|---|
| 1 (Easy) | BERT multi-label tag classifier | MagnaTagATune, top-50 tags | `src/bert_encoder.py` |
| 2 (Medium) | GraphSAGE / GAT on structure graphs | GTZAN, 10 genres | `src/gnn_model.py` |
| 3 (Hard) | GNN–BERT fusion, multi-task | MagnaTagATune + DEAM | `src/fusion_model.py` |
| 4 (Advanced) | Contrastive dual encoder, InfoNCE | MusicCaps (MTAT fallback) | `src/contrastive.py` |

---

## 1. Setup

**Check your interpreter first.** A machine often has several Python installs
and only one has `torch`; running under the wrong one gives an `ImportError`
that looks like a missing package rather than a wrong interpreter.

```bash
python tools/check_env.py
```

```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the reported results were produced
with. For a GPU, install the CUDA torch wheel first, then the rest.

---

## 2. Quickstart

Datasets are not in the repository — fetch what you need:

```bash
python tools/fetch_data.py --list
```

```bash
python tools/fetch_data.py magnatagatune --extract
```

Then preprocess and train, using the per-corpus preset:

```bash
python src/preprocess.py --config configs/mtat.yaml
```

```bash
python src/train.py all --config configs/mtat.yaml
```

```bash
python src/evaluate.py --set paths.results=results_mtat --set paths.plots=results_mtat/plots
```

**CPU-only is slow** — the BERT-based tasks take 6–8 h for the full suite.
`notebooks/colab_run.ipynb` runs everything on a free Colab T4 in roughly
40–70 min, downloads the corpora inside Colab, and hands back a zip of
`results_*/`. That is the intended path.

Individual tasks:

```bash
python src/train.py task1 --config configs/mtat.yaml
python src/train.py task2 --config configs/gtzan.yaml --set gnn.conv=gat
python src/train.py task3 --config configs/mtat.yaml --ablation
python src/train.py task4 --config configs/mtat.yaml
```

Any config key can be overridden from the command line:

```bash
python src/train.py task3 --config configs/mtat.yaml --set train.epochs=40 --set gnn.hidden_dim=256
```

Tests:

```bash
python tests/test_pipeline.py
```

---

## 3. Data

### 3.1 Corpora

The spec requires at least one primary audio dataset and one text/tag source.
**MagnaTagATune satisfies both** (29 s audio + 188 human tags), so it carries
Tasks 1, 3 and 4; GTZAN carries Task 2's genre classification; DEAM supplies the
auxiliary emotion target.

| `dataset.source` | Size | Role | Fetch |
|---|---|---|---|
| `magnatagatune` | 2.8 GB, 25,877 clips, 188 tags | Tasks 1, 3, 4 | `fetch_data.py magnatagatune --extract` |
| `gtzan` | 1.1 GB, 1,000 clips, 10 genres | Task 2 | `fetch_data.py gtzan --extract` |
| `deam` | 1.7 GB, 1,802 clips, valence/arousal | `L_aux` | `fetch_data.py deam --extract` |
| `musiccaps` | 0.3 GB, 5,521 clips + captions | Task 4 | `fetch_data.py musiccaps --clips 1200` |
| `fma_small` / `fma_medium` | 7.7 / 22 GB | Task 2 alternative | `fetch_data.py fma_small --extract` |
| `synthetic` | — | pipeline sanity check | built in |

Per-corpus presets live in `configs/`. They are **overlays** on `config.yaml`
(via an `extends:` key), so they cannot drift from the base config.

**MusicCaps is best-effort.** Its audio is not redistributable — it ships as
YouTube ids, so each clip is pulled with `yt-dlp` (needs `ffmpeg`). Videos
disappear and YouTube rate-limits, so the fetcher keeps a resume manifest,
classifies permanent vs transient failures, reports its success rate, and stops
early if the rate collapses. If too few clips land, run Task 4 on
MagnaTagATune and say so in the report — a documented deviation costs far less
than a table built on 40 clips.

### 3.2 A text branch that cannot cheat

MagnaTagATune and GTZAN have **no captions**: their only text *is* the tag set.
Feeding those tags to BERT and scoring against the same tags measures nothing —
on GTZAN, Task 1 reached validation Macro-F1 **1.0000 in a single epoch**.

`dataset.text_tag_fraction` fixes this by partitioning the tag vocabulary into
two **disjoint** halves: that fraction becomes the text the encoder sees, and
the remainder is the prediction target. The text is then genuinely informative
without containing the answer. At `0.0` the text restates the target and Task 1
saturates meaninglessly.

**Disjoint per-tag is not enough.** MTAT has `male`, `male vocal` and
`male voice` as three separate tags, so a per-tag random split puts one in the
text and another in the target — and the text then literally contains a word of
its own answer. Measured on a 2,500-clip subsample: **156 of 2,499 records
(6%) leaked**. `dataset.cluster_tags` therefore groups tags into synonym
clusters (shared token after singularisation, plus a short exact-synonym list,
chained with union-find) and moves whole clusters. On the top-50 vocabulary
that gives 32 clusters and correctly pulls all 15 vocal-related tags together.
Preprocessing then **asserts** no target tag appears in any record's text, so
this cannot regress silently. Leaks: 156 → 0.

The clustering is lexical, not semantic — `techno` on the text side still
correlates with `electronic` on the target side. Entailment-based clustering
collapses most of the vocabulary into one blob, so that residual correlation is
accepted and reported as a limitation.

The synthetic corpus has the analogous knob, `dataset.label_dropout` (default
`0.5`): captions hedge their genre/mood word half the time, because real
captions describe a clip without restating its tag set.

### 3.3 Splits and leakage

Splits are **artist-grouped** and **label-stratified**, and preprocessing
**raises** on leakage rather than warning. This matters: a 2,500-clip MTAT
subsample spans ~230 artists, and clips from one artist are often successive
excerpts of the same recording, so an ungrouped split lets the model memorise a
track and be scored on its neighbours.

MagnaTagATune's *official* split (by audio directory) is supported for
comparability with published numbers, but it shares artists across partitions —
the pipeline reports exactly how many. Use `--set split.use_official=false` for
the artist-disjoint split.

---

## 4. Preprocessing

1. **Audio** — resample to 22,050 Hz; log-mel (128), chroma (12), MFCC (20);
   normalised per track.
2. **Segmentation** — fixed windows, or beat-synchronous via `librosa.beat`
   (`audio.beat_sync: true`).
3. **Node features** — mean+std pooling per segment (308 dims).
4. **Graphs** — `graph.kind` selects:
   - `segment` — nodes are time segments; edges are temporal adjacency plus
     MFCC/chroma cosine similarity above `τ`, with a k-NN connectivity backstop.
   - `chord` — nodes are unique chords (chroma matched against 24 major/minor
     triad templates); edges are transitions weighted by count, kept
     **directed** because a progression is not symmetric.
   - `hybrid` — both, linked segment ↔ its recognised chord.
5. **Text** — BERT tokenisation, 128 tokens.
6. **Splits** — as above.

Every edge carries an `edge_type` (temporal / similarity / transition / self),
which the Task 3 case studies read back out.

### Graph density is a correctness issue, not a tuning knob

A 29–30 s clip cut into 5 s windows is a **six-node graph**, and with a fixed
k-NN backstop of 4 nearly every pair gets wired: measured density **0.95**, an
almost complete graph. Message passing over a complete graph is close to mean
pooling — so Tasks 2 and 3 lose the structural premise that motivates them
while still *appearing* to train fine.

Two changes, both reflected in the defaults:

- `graph_builder.effective_knn` scales the backstop with graph size. Its job is
  to prevent isolated nodes, which k=1 already does; it widens only once the
  graph is big enough (e.g. beat-synchronous segmentation, 30–70 nodes) for a
  larger k to mean anything.
- `audio.segment_seconds: 2.5` for ~30 s clips → 12 nodes instead of 6.

Measured result: density **0.27–0.31**, no isolated nodes, and similarity edges
that correspond to genuinely recurring material. **If you change the
segmentation, re-check density in `notebooks/eda.ipynb` before trusting any
Task 2/3 number.**

---

## 5. Models

**Task 1** — `t = BERT_CLS(X_text)`, per-tag sigmoid, BCE. Full fine-tuning, a
frozen encoder, or unfreezing the last *N* blocks (`text.unfreeze_last_n`),
which dominates CPU runtime.

**Task 2** — GraphSAGE `h_i^(l+1) = σ(W^(l)·CONCAT(h_i^(l), MEAN_{j∈N(i)} h_j^(l)))`
or GAT, with LayerNorm, residuals, and mean (or mean+max) readout. Node features
are **audio-only**, which is what makes the Task 1 / Task 2 / Task 3 comparison
interpretable.

**Task 3** — cross-attention with the graph vector as query:

```
A = softmax(QKᵀ/√d),  Q = g·W_Q,  K = H_text·W_K
z = CONCAT(g, A·H_text)
L = L_tags + α‖v − v̂‖² + β‖a − â‖²
```

`fusion.mode` selects the ablation arm — `cross_attention`, `concat`,
`bert_only`, `gnn_only` — so all four arms are the same code path with the same
schedule and seed. The emotion term is **masked**, so corpora without
valence/arousal contribute only the tagging term.

**Task 4** — symmetric InfoNCE over in-batch negatives, retrieval in both
directions, plus zero-shot tagging by scoring each graph against a per-tag
prompt. Symmetrising matters: trained one-directionally, retrieval collapses in
the other direction.

---

## 6. Evaluation

`src/metrics.py` implements spec §6: per-tag precision/recall/F1, Macro-F1,
Micro-F1, mean AUC-PR, MAE/R² for emotion, R@K / median rank / MRR for
retrieval, and the graph coherence score.

Four methodological points, because they change the numbers:

- **Thresholds are tuned on validation, never on test.** A flat 0.5 cut is a
  poor operating point for sparse multi-label targets, but tuning on test
  inflates every model — and inflates the *random* baseline most. This was
  caught here: a random predictor briefly scored 0.77 Macro-F1, matching the
  trained models; scored correctly it reaches 0.41.
- **Macro-F1 excludes tags with no positives in the split.** F1 is undefined
  there, and averaging in a zero deflates the score by an amount that depends on
  the split rather than the model. `macro_f1_all_tags` keeps the zero-filled
  variant.
- **`pos_weight` is capped.** Uncapped, rare tags get weights in the hundreds
  and dominate the gradient.
- **The spec's coherence score is uninformative as stated.** `S_graph` saturates
  at ≈0.999 for every graph, every depth, and for non-edges too — node features
  come out of a ReLU, so cosine similarity between *any* two vectors is high.
  It measures the activation function, not the model. `graph_coherence_score`
  therefore also reports the **lift** of edge similarity over non-edge
  similarity from the same graph, which is the informative quantity.

---

## 7. Results and provenance

Every metrics row carries a `_provenance` block (corpus, label space, graph
kind, segment length, split). `merge_metrics` **raises** rather than mixing
corpora in one file, and `evaluate.py` refuses to build a mixed table. See
[results/README.md](results/README.md) for what each file holds.

Score each corpus into its own results directory:

```bash
python src/train.py all --config configs/gtzan.yaml   # -> results_gtzan/
```

---

## 8. Human evaluation (Task 4)

Spec §6 requires ≥5 listeners rating caption↔clip match on a 1–5 scale.

```bash
python tools/make_human_eval.py --results results_mtat --queries 10
```

This exports the retrieved clips and writes **one self-contained HTML file**
(audio embedded, no server or install needed). Send it to at least 5 people;
each downloads a `ratings_*.json` when done. Put those in
`results_mtat/human_eval/ratings/` and aggregate:

```bash
python tools/aggregate_human_eval.py --results results_mtat
```

Study design, because it decides whether the numbers mean anything: ratings are
per (caption, clip) pair with the **retrieval rank hidden**, order shuffled per
rater, and one **distractor** from a different query planted in each question.
The distractor is a check on the raters — if it does not score clearly lowest,
the ratings are noise and the script says so. Agreement is reported as
Krippendorff's α for ordinal data.

---

## 9. Repository layout

```
gnn-bert-music-context/
├── config.yaml               base config; --set overrides any key
├── configs/                  per-corpus overlays (extends: ../config.yaml)
├── requirements.txt          exact pinned versions
├── data/{raw,processed*,splits*}
├── notebooks/
│   ├── eda.ipynb             corpus, graphs, chord recognition, split integrity
│   ├── demo_context.ipynb    end-to-end single-track inference
│   └── colab_run.ipynb       the GPU run
├── src/
│   ├── audio_features.py     mel, chroma, MFCC, segmentation
│   ├── graph_builder.py      chord + segment + hybrid graphs
│   ├── bert_encoder.py       Task 1 and the shared text branch
│   ├── gnn_model.py          GraphSAGE / GAT + readout + coherence
│   ├── fusion_model.py       cross-attention fusion, multi-task loss
│   ├── contrastive.py        Task 4 InfoNCE, retrieval, zero-shot
│   ├── baselines.py          B1 prior/random, B2 mel-CNN, B4 PCA+MLP
│   ├── metrics.py            all evaluation metrics
│   ├── engine.py             shared training loop, checkpoint guards
│   ├── preprocess.py / train.py / evaluate.py    CLIs
│   └── data/                 adapters, synthetic corpus, dataset
├── tests/test_pipeline.py    33 tests
├── tools/
│   ├── fetch_data.py         resumable corpus downloader
│   ├── check_env.py          interpreter / dependency diagnosis
│   ├── export_deliverables.py  regenerate exports from a checkpoint
│   ├── make_human_eval.py / aggregate_human_eval.py
│   ├── make_notebooks.py     notebooks are generated, not hand-edited
│   └── make_report_tables.py LaTeX tables from metrics.json
└── report/                   main.tex + generated tables
```

---

## 10. Baselines

- **B1** — random and tag-prior predictors
- **B2** — CNN on the mel-spectrogram, consuming the *same* node features as the
  GNN laid out as a (segments × features) image, so the B2/Task 2 gap isolates
  relational structure rather than feature engineering
- **B3** — BERT-only (Task 1, and the `bert_only` fusion arm)
- **B4** — PCA + MLP on mean/std-pooled features

---

## 11. Reproducibility

- `seed` in `config.yaml` seeds Python, numpy and torch.
- Splits are deterministic given the seed and asserted leak-free.
- Every run records its best epoch and the threshold it was scored at.
- Checkpoints carry their corpus fingerprint; loading one against a different
  label space raises `CheckpointMismatch` with an explanation instead of a bare
  tensor-shape error.
- Notebooks are generated by `tools/make_notebooks.py` — edit that, not the
  `.ipynb`.
- `distilbert-base-uncased` downloads from the HuggingFace hub on first run.
- Runs are **single-seed**; small gaps between ablation arms should not be
  over-read.

---

## 12. Use of AI assistance

Parts of this project — pipeline code, experiment scripts and drafting of the
report — were developed with the assistance of Claude (Anthropic). All
experiments were run, and all reported results checked against the generated
metrics files, by the authors.
