"""Build the Task 4 human-evaluation study (spec section 6).

    "Human evaluation (Task 4). Minimum 5 listeners rate whether retrieved clip
     matches caption on scale [1,5]."

This turns the retrieval output into a study a person can actually sit down and
do: it exports the retrieved audio, then writes ONE self-contained HTML file
with the clips embedded, so a rater needs no server, no install, and no network
-- just a browser. They click through, hit "Download my ratings", and send back
a small JSON file. `aggregate_human_eval.py` turns those files into the report
table.

    python tools/make_human_eval.py --results results_mtat --queries 10
    # -> results_mtat/human_eval/rating_form.html   (open in any browser)

Design notes that matter for the study being worth anything:

* Ratings are collected per (caption, clip) PAIR, not per query, so a rater
  scores the model's top-3 without being told which is the "correct" one.
* Presentation order is shuffled per rater, and rank is not shown, so a rater
  cannot infer the model's confidence and anchor on it.
* One distractor -- a clip drawn from a different query -- is mixed into each
  query's candidates. It is a sanity check on the raters themselves: if the
  distractor does not score clearly lowest on average, the ratings are noise.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from utils import load_config, load_json, resolve, save_json  # noqa: E402


def _relocate_index(corpus: str, raw_root: Path) -> dict:
    """Rebuild track_id -> local audio path by re-reading the raw corpus.

    `index.json` stores the absolute path the preprocessing run saw. Training
    happened on Kaggle, so those read /kaggle/working/... and resolve to nothing
    on the machine building the study. Re-deriving them from the corpus on this
    disk is what lets the listening study be built anywhere the audio exists.
    """
    from data.adapters import READERS

    reader = READERS.get(corpus)
    if reader is None:
        return {}
    try:
        kwargs = {"top_tags": 50} if corpus == "magnatagatune" else {}
        return {r["track_id"]: Path(r["audio_path"])
                for r in reader(Path(raw_root), **kwargs) if r.get("audio_path")}
    except Exception as exc:  # noqa: BLE001 - a missing corpus is not fatal here
        print(f"  could not re-read {corpus} from {raw_root}: "
              f"{type(exc).__name__}: {exc}")
        return {}


def find_audio(track_id: str, index: list[dict],
               fallback: dict | None = None) -> Path | None:
    """Locate a track's source audio from the preprocessing index."""
    for record in index:
        if record["track_id"] == track_id:
            path = record.get("audio_path")
            if path and Path(path).exists():
                return Path(path)
            break
    if fallback:
        candidate = fallback.get(track_id)
        if candidate and candidate.exists():
            return candidate
    return None


def export_clip(src: Path, dst: Path, seconds: float = 12.0,
                sample_rate: int = 16000) -> bool:
    """Write a short excerpt as a small mono file the browser can play.

    16 kHz mono is deliberate: a rater is judging whether a clip matches a
    description, not audio quality, and an uncompressed 22 kHz stereo excerpt
    triples the size of a form that has to be emailed. At 16 kHz/12 s a clip is
    ~384 KB, so a 40-clip embedded form lands around 20 MB.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        from audio_features import load_audio

        y = load_audio(src, sample_rate=sample_rate, duration=seconds)
        sf.write(dst.with_suffix(".wav"), y, sample_rate, subtype="PCM_16")
        return True
    except Exception as exc:  # noqa: BLE001 - one unreadable clip must not stop the build
        print(f"    [skip] {src.name}: {type(exc).__name__}: {exc}")
        return False


def data_uri(path: Path) -> str:
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(path.suffix.lstrip(".").lower(),
                                                        "audio/wav")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Music retrieval listening study</title>
<style>
  :root {
    --bg: #fbfaf8; --fg: #1c1b19; --muted: #6b6862;
    --line: #e2ded7; --card: #ffffff; --accent: #3f6f52;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 760px; margin: 0 auto; padding: 32px 20px 120px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 28px; }
  .intro { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
           padding: 18px 20px; margin-bottom: 28px; }
  .intro ol { margin: 8px 0 0; padding-left: 20px; }
  .intro li { margin: 4px 0; }
  label.name { display: block; font-weight: 600; margin: 14px 0 4px; }
  input[type=text] { width: 100%; padding: 9px 11px; font-size: 1rem;
                     border: 1px solid var(--line); border-radius: 7px; background: #fff; }
  .q { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
       padding: 20px; margin-bottom: 18px; }
  .qhead { font-size: 0.78rem; letter-spacing: .08em; text-transform: uppercase;
           color: var(--muted); margin-bottom: 8px; }
  .caption { font-size: 1.05rem; font-style: italic; margin: 0 0 18px;
             padding-left: 12px; border-left: 3px solid var(--accent); }
  .cand { padding: 14px 0; border-top: 1px solid var(--line); }
  audio { width: 100%; margin: 6px 0 10px; }
  .scale { display: flex; gap: 6px; flex-wrap: wrap; }
  .scale label { flex: 1 1 auto; min-width: 108px; text-align: center; cursor: pointer;
                 border: 1px solid var(--line); border-radius: 7px; padding: 7px 4px;
                 font-size: 0.85rem; background: #fff; }
  .scale input { margin-right: 5px; }
  .scale label:has(input:checked) { border-color: var(--accent);
                                    box-shadow: inset 0 0 0 1px var(--accent);
                                    background: #f1f6f2; }
  .bar { position: fixed; left: 0; right: 0; bottom: 0; background: var(--card);
         border-top: 1px solid var(--line); padding: 12px 20px;
         display: flex; align-items: center; gap: 16px; justify-content: center; }
  button { background: var(--accent); color: #fff; border: 0; border-radius: 8px;
           padding: 11px 20px; font-size: 0.95rem; cursor: pointer; }
  button[disabled] { background: #b3b0aa; cursor: not-allowed; }
  .count { color: var(--muted); font-size: 0.9rem; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#171614; --fg:#ece9e4; --muted:#9c978e; --line:#33302b;
            --card:#201f1c; --accent:#7fb694; }
    input[type=text], .scale label { background:#1a1917; color:var(--fg); }
    .scale label:has(input:checked) { background:#24302a; }
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>Music retrieval listening study</h1>
  <p class="sub">Task 4 evaluation &middot; about 15 minutes</p>

  <div class="intro">
    <strong>What to do</strong>
    <ol>
      <li>Read the description, then listen to each clip under it.</li>
      <li>Rate how well <em>that clip</em> matches <em>that description</em>, 1&ndash;5.</li>
      <li>Judge each clip on its own. The clips are in no meaningful order, and
          there is not necessarily exactly one good match &mdash; there may be
          several, or none.</li>
      <li>When every clip is rated, click <strong>Download my ratings</strong>
          and send back the file.</li>
    </ol>
    <label class="name" for="rater">Your name or initials</label>
    <input type="text" id="rater" placeholder="e.g. A.R." autocomplete="off">
  </div>

  <div id="questions"></div>
</div>

<div class="bar">
  <span class="count" id="count"></span>
  <button id="save" disabled>Download my ratings</button>
</div>

<script>
const STUDY = __STUDY_JSON__;
const SCALE = [
  [1, "1 - no match"],
  [2, "2 - poor"],
  [3, "3 - partial"],
  [4, "4 - good"],
  [5, "5 - excellent"]
];

// Shuffle query order and, within a query, candidate order -- per rater, seeded
// by the clock. Rank is never shown, so a rater cannot anchor on the model's
// own confidence ordering.
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const order = shuffle(STUDY.questions.map((_, i) => i));
const container = document.getElementById("questions");
let total = 0;

order.forEach((qi, shown) => {
  const q = STUDY.questions[qi];
  const div = document.createElement("div");
  div.className = "q";
  const head = document.createElement("div");
  head.className = "qhead";
  head.textContent = `Description ${shown + 1} of ${STUDY.questions.length}`;
  div.appendChild(head);

  const cap = document.createElement("p");
  cap.className = "caption";
  cap.textContent = q.caption;
  div.appendChild(cap);

  shuffle(q.candidates.slice()).forEach(c => {
    total += 1;
    const wrap = document.createElement("div");
    wrap.className = "cand";

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = c.audio;
    wrap.appendChild(audio);

    const scale = document.createElement("div");
    scale.className = "scale";
    SCALE.forEach(([value, text]) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = `q${q.query_id}_c${c.candidate_id}`;
      input.value = value;
      input.dataset.query = q.query_id;
      input.dataset.candidate = c.candidate_id;
      input.addEventListener("change", refresh);
      label.appendChild(input);
      label.appendChild(document.createTextNode(text));
      scale.appendChild(label);
    });
    wrap.appendChild(scale);
    div.appendChild(wrap);
  });
  container.appendChild(div);
});

function collected() {
  return Array.from(document.querySelectorAll("input[type=radio]:checked"));
}

function refresh() {
  const n = collected().length;
  document.getElementById("count").textContent = `${n} of ${total} clips rated`;
  document.getElementById("save").disabled = n < total;
}

document.getElementById("save").addEventListener("click", () => {
  const rater = (document.getElementById("rater").value || "anonymous").trim();
  const ratings = collected().map(el => ({
    query_id: Number(el.dataset.query),
    candidate_id: Number(el.dataset.candidate),
    score: Number(el.value)
  }));
  const blob = new Blob([JSON.stringify({
    study_id: STUDY.study_id,
    rater: rater,
    submitted_at: new Date().toISOString(),
    ratings: ratings
  }, null, 2)], { type: "application/json" });

  const safe = rater.replace(/[^A-Za-z0-9._-]+/g, "_") || "anonymous";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ratings_${safe}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
});

refresh();
</script>
</body>
</html>
"""


def build(results_dir: Path, processed_dir: Path, n_queries: int,
          seconds: float, seed: int, embed: bool = True,
          sample_rate: int = 16000) -> Path:
    retrieval = results_dir / "retrieval_examples" / "caption_to_audio.json"
    if not retrieval.exists():
        raise SystemExit(f"missing {retrieval}\nRun Task 4 first:  "
                         f"python src/train.py task4")
    examples = load_json(retrieval)
    index = load_json(processed_dir / "index.json")

    # If the recorded paths came from another machine (a GPU run on Kaggle or
    # Colab), re-derive them from the corpus on this disk.
    meta_path = Path(processed_dir) / "meta.json"
    corpus = load_json(meta_path).get("source") if meta_path.exists() else None
    sample = next((r.get("audio_path") for r in index if r.get("audio_path")), None)
    fallback = {}
    if sample and not Path(sample).exists():
        print(f"  recorded audio paths do not resolve here ({sample[:48]}...)")
        print(f"  re-locating {corpus} audio under {REPO / 'data' / 'raw'}")
        fallback = _relocate_index(corpus, REPO / "data" / "raw")
        print(f"  re-located {len(fallback)} tracks")

    out_dir = results_dir / "human_eval"
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # An answer key, kept OUT of the HTML: the rater must not be able to read
    # which candidate was the ground-truth pair or what rank it held.
    questions, key = [], []
    chosen = examples[:n_queries]

    for qi, ex in enumerate(chosen):
        candidates = []
        pool = [(r["track_id"], r["rank"], r["is_correct"]) for r in ex["results"]]

        # Include the TRUE pair even when the model failed to retrieve it.
        #
        # At R@1 ~ 0.01 over ~600 candidates the correct clip is almost never in
        # the top 3, so a study built from retrieved clips alone contains no
        # correct answer at all: raters hear only wrong ones, every score is
        # low, and the numbers say nothing about the model. Adding the true pair
        # gives the study a ceiling to measure against -- "how much worse than
        # correct are the retrieved clips" -- and is what makes the
        # ground-truth / retrieved / distractor comparison interpretable.
        # It is marked only in the answer key; in the form it is indistinguishable.
        if not any(is_correct for _, _, is_correct in pool):
            pool.append((ex["ground_truth"], 0, True))

        # One distractor from a different query: a check on rater attention.
        others = [e for j, e in enumerate(chosen) if j != qi]
        if others:
            far = rng.choice(others)
            pool.append((far["ground_truth"], -1, False))

        for ci, (track_id, rank, is_correct) in enumerate(pool):
            src = find_audio(track_id, index, fallback)
            if src is None:
                print(f"    [skip] no audio for {track_id}")
                continue
            stem = clip_dir / f"q{qi:02d}_c{ci}"
            if not export_clip(src, stem, seconds, sample_rate):
                continue
            written = next((p for p in (stem.with_suffix(".wav"), stem.with_suffix(".mp3"))
                            if p.exists()), None)
            if written is None:
                continue
            # Embedded: one file a rater can just open. External: a folder
            # to zip, for when the embedded form is too big to email.
            audio_ref = data_uri(written) if embed else f"clips/{written.name}"
            candidates.append({"candidate_id": ci, "audio": audio_ref})
            key.append({"query_id": qi, "candidate_id": ci, "track_id": track_id,
                        "model_rank": rank, "is_ground_truth": bool(is_correct),
                        "is_distractor": rank == -1})

        if candidates:
            questions.append({"query_id": qi, "caption": ex["query_caption"],
                              "candidates": candidates})

    if not questions:
        raise SystemExit("no clips could be exported -- is the raw audio still in data/raw?")

    study = {"study_id": f"task4-{results_dir.name}", "questions": questions}
    html = HTML.replace("__STUDY_JSON__", json.dumps(study))
    form = out_dir / "rating_form.html"
    form.write_text(html, encoding="utf-8")

    save_json(key, out_dir / "answer_key.json")
    save_json({"study_id": study["study_id"],
               "n_queries": len(questions),
               "n_pairs": sum(len(q["candidates"]) for q in questions),
               "clip_seconds": seconds,
               "note": "answer_key.json is deliberately not embedded in the HTML"},
              out_dir / "study_meta.json")

    n_pairs = sum(len(q["candidates"]) for q in questions)
    size_mb = form.stat().st_size / 1e6
    print(f"\n  {len(questions)} descriptions x ~{n_pairs // max(len(questions), 1)} clips "
          f"= {n_pairs} ratings per person")
    print(f"  form      : {form}  ({size_mb:.1f} MB, self-contained)")
    print(f"  answer key: {out_dir / 'answer_key.json'}  (do not send to raters)")
    print("\n  Send rating_form.html to at least 5 listeners. Collect their")
    print(f"  ratings_*.json into {out_dir / 'ratings'}/ then run:")
    print(f"    python tools/aggregate_human_eval.py --results {results_dir.name}")
    return form


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Task 4 listening study.")
    ap.add_argument("--results", default="results",
                    help="results dir holding retrieval_examples/ (e.g. results_mtat)")
    ap.add_argument("--processed", default=None,
                    help="processed dir holding index.json (default: from config)")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="excerpt length; keep short so the form stays small")
    ap.add_argument("--sample-rate", type=int, default=16000,
                    help="excerpt sample rate; 16 kHz is ample for judging a match")
    ap.add_argument("--external", action="store_true",
                    help="reference clips/ as files instead of embedding them "
                         "(smaller HTML; send the zipped folder)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    results_dir = Path(args.results)
    if not results_dir.is_absolute():
        results_dir = REPO / results_dir
    processed = Path(args.processed) if args.processed else resolve(cfg, "processed")
    if not processed.is_absolute():
        processed = REPO / processed

    build(results_dir, processed, args.queries, args.seconds,
          cfg.get("seed", 42), embed=not args.external,
          sample_rate=args.sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
