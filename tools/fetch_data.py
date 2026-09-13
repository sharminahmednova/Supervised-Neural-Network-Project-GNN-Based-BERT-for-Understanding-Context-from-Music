"""Resumable downloader for the corpora in the spec's Table 1.

    python tools/fetch_data.py fma_metadata
    python tools/fetch_data.py fma_small --extract
    python tools/fetch_data.py deam --extract
    python tools/fetch_data.py --list

Why this exists rather than a curl one-liner: curl on Windows uses schannel,
the native TLS stack, which raises

    curl: (56) schannel: server closed abruptly (missing close_notify)

whenever a server tears down a long transfer without TLS's close_notify. The
bytes already received are fine, but bare curl gives up. This goes through
Python's OpenSSL instead, resumes with an HTTP Range request, and retries with
backoff -- so a 7 GB download survives being dropped a dozen times.

Nothing is downloaded unless you name a dataset, and every file lands under
data/raw/<dataset>/ in exactly the layout src/data/adapters.py expects.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"

FMA_BASE = "https://os.unil.cloud.switch.ch/fma"
DEAM_BASE = "https://cvml.unige.ch/databases/DEAM"
# GTZAN's original host (marsyas.info) has been dead for years. The marsyas org
# mirrors the dataset on the HuggingFace hub, which is the only reliably live
# source found: the opihi.cs.waikato.ac.nz mirror no longer resolves.
GTZAN_URL = ("https://huggingface.co/datasets/marsyas/gtzan/"
             "resolve/main/data/genres.tar.gz")

# dest_dir is relative to data/raw/; the archives already carry their own
# top-level folder (fma_small/, fma_metadata/), so they extract in place.
MTAT_BASE = "https://mirg.city.ac.uk/datasets/magnatagatune"

# MagnaTagATune ships its audio as a split zip (mp3.zip.001/.002/.003) that has
# to be concatenated before it can be opened -- handled by fetch_magnatagatune.
MTAT_PARTS = [f"{MTAT_BASE}/mp3.zip.{i:03d}" for i in (1, 2, 3)]
MTAT_ANNOTATIONS = f"{MTAT_BASE}/annotations_final.csv"

DATASETS = {
    "gtzan": {
        "url": GTZAN_URL,
        "dest": "gtzan",
        "approx_gb": 1.14,
        "note": "1,000 clips / 30 s / 10 genres. Smallest real corpus; .au audio.",
    },
    "fma_metadata": {
        "url": f"{FMA_BASE}/fma_metadata.zip",
        "dest": "fma",
        "approx_gb": 0.36,
        "note": "tracks.csv, genres.csv, official train/val/test splits. Required for any FMA run.",
    },
    "fma_small": {
        "url": f"{FMA_BASE}/fma_small.zip",
        "dest": "fma",
        "approx_gb": 7.68,
        "note": "8,000 clips / 30 s / 8 balanced genres.",
    },
    "fma_medium": {
        "url": f"{FMA_BASE}/fma_medium.zip",
        "dest": "fma",
        "approx_gb": 22.19,
        "note": "25,000 clips / 16 genres. Large -- only worth it with a GPU.",
    },
    "deam_annotations": {
        "url": f"{DEAM_BASE}/DEAM_Annotations.zip",
        "dest": "deam",
        "approx_gb": 0.03,
        "note": "valence/arousal annotations (static + dynamic).",
    },
    "deam_audio": {
        "url": f"{DEAM_BASE}/DEAM_audio.zip",
        "dest": "deam",
        "approx_gb": 1.7,
        "note": "1,802 excerpts, 45 s MP3.",
    },
}
ALIASES = {
    "fma": ["fma_metadata", "fma_small"],
    "deam": ["deam_annotations", "deam_audio"],
}

CHUNK = 1 << 20  # 1 MiB


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def remote_size(url: str, timeout: int = 30) -> int | None:
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError, OSError):
        return None


def download(url: str, out: Path, max_retries: int = 40, timeout: int = 60) -> Path:
    """Download with HTTP Range resume and exponential backoff."""
    out.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    if total:
        print(f"  remote size: {_human(total)}")
    else:
        print("  remote size: unknown (server gave no Content-Length)")

    attempt = 0
    stall_count = 0
    while True:
        have = out.stat().st_size if out.exists() else 0
        if total and have >= total:
            print(f"  complete: {out.name} ({_human(have)})")
            return out

        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
            print(f"  resuming at {_human(have)}"
                  + (f" / {_human(total)} ({100*have/total:.1f}%)" if total else ""))

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # 200 to a Range request means the server ignored it: restart.
                mode = "ab"
                if have and resp.status == 200:
                    print("  server ignored Range -- restarting from 0")
                    mode, have = "wb", 0
                start = time.time()
                written = 0
                with open(out, mode) as fh:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        written += len(block)
                        done = have + written
                        elapsed = max(time.time() - start, 1e-6)
                        rate = written / elapsed / 1048576
                        if total:
                            pct = 100 * done / total
                            eta = (total - done) / max(written / elapsed, 1)
                            sys.stdout.write(
                                f"\r  {_human(done)} / {_human(total)}  "
                                f"{pct:5.1f}%  {rate:5.1f} MB/s  ETA {eta/60:5.1f} min   ")
                        else:
                            sys.stdout.write(f"\r  {_human(done)}  {rate:5.1f} MB/s   ")
                        sys.stdout.flush()
                sys.stdout.write("\n")

            if written == 0:
                stall_count += 1
                if stall_count >= 3:
                    raise RuntimeError(
                        f"server returned no data three times for {url}; giving up")
            else:
                stall_count = 0

        except KeyboardInterrupt:
            print(f"\n  interrupted -- {_human(out.stat().st_size if out.exists() else 0)} "
                  f"kept; re-run to resume")
            raise
        except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"gave up after {max_retries} retries: {exc}") from exc
            wait = min(2 ** min(attempt, 6), 60)
            print(f"\n  [{attempt}/{max_retries}] {type(exc).__name__}: {exc}"
                  f"  -- retrying in {wait}s")
            time.sleep(wait)
            continue

        if not total:  # no Content-Length: one clean pass is all we can verify
            print(f"  finished: {out.name} ({_human(out.stat().st_size)})")
            return out


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def extract(archive: Path, dest: Path, delete_after: bool = False) -> None:
    """Extract a .zip or .tar.gz, refusing entries that escape the destination."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    print(f"  extracting {archive.name} -> {dest}")

    def unsafe(name: str) -> bool:
        return not str((root / name).resolve()).startswith(str(root))

    try:
        if archive.name.endswith((".tar.gz", ".tgz", ".tar")):
            import tarfile

            mode = "r:gz" if archive.name.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(archive, mode) as tf:
                members = tf.getmembers()
                for m in members:
                    if unsafe(m.name) or m.issym() or m.islnk():
                        raise RuntimeError(f"unsafe entry in archive: {m.name}")
                tf.extractall(dest)
        else:
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    if unsafe(member.filename):
                        raise RuntimeError(f"unsafe path in archive: {member.filename}")
                zf.extractall(dest)
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise SystemExit(
            f"  {archive.name} could not be extracted ({type(exc).__name__}: {exc}).\n"
            f"  The download is probably incomplete -- re-run without --extract "
            f"to resume it.") from exc
    print("  extracted")
    if delete_after:
        archive.unlink()
        print(f"  removed {archive.name}")


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1073741824


def fetch_magnatagatune(do_extract: bool, keep_zip: bool) -> None:
    """Annotations + the three-part split zip, joined before extraction."""
    dest = RAW / "magnatagatune"
    dest.mkdir(parents=True, exist_ok=True)
    print("\n== magnatagatune ==  ~2.8 GB")
    print("  25,877 clips / 29 s, 188 human tags. Audio is a 3-part split zip.")
    print(f"  disk free: {disk_free_gb(RAW):.1f} GB   needed: ~6 GB "
          f"(parts + joined zip + extracted)")

    download(MTAT_ANNOTATIONS, dest / "annotations_final.csv")

    parts = []
    for url in MTAT_PARTS:
        part = dest / Path(url).name
        download(url, part)
        parts.append(part)

    if not do_extract:
        print("  parts downloaded; re-run with --extract to join and unpack")
        return

    joined = dest / "mp3.zip"
    expected = sum(p.stat().st_size for p in parts)
    if not (joined.exists() and joined.stat().st_size == expected):
        print(f"  joining {len(parts)} parts -> mp3.zip ({_human(expected)})")
        with open(joined, "wb") as out:
            for p in parts:
                with open(p, "rb") as fh:
                    shutil.copyfileobj(fh, out, CHUNK)

    extract(joined, dest, delete_after=True)
    if not keep_zip:
        for p in parts:
            p.unlink()
        print("  removed split parts")

    n = len(list(dest.rglob("*.mp3")))
    print(f"  {n} mp3 files present (expect ~25,863)")


MUSICCAPS_CSV = ("https://huggingface.co/datasets/google/MusicCaps/"
                 "resolve/main/musiccaps-public.csv")


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def fetch_musiccaps(n_clips: int = 1200, seed: int = 42) -> None:
    """Captions from HuggingFace, audio from YouTube one clip at a time.

    MusicCaps ships captions and YouTube ids, not audio -- the audio is not
    redistributable. So each 10 s clip has to be pulled individually with
    yt-dlp, which is slow and regularly rate-limited or blocked, and some videos
    are simply gone. That makes this fundamentally best-effort:

    * a manifest records the outcome per id, so re-running resumes rather than
      restarting;
    * failures are counted and reported, never fatal;
    * `src/data/adapters.py::read_musiccaps` already skips rows whose audio is
      absent, so a partial download trains without any further change.

    If too few clips land, run Task 4 on MagnaTagATune instead and say so in the
    report -- a documented deviation costs less than a fabricated table.
    """
    import csv as _csv
    import json as _json
    import random as _random
    import subprocess

    dest = RAW / "musiccaps"
    audio_dir = dest / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print("\n== musiccaps ==")
    print("  5,521 clips / 10 s with expert captions. Audio is NOT redistributable:")
    print("  each clip is fetched from YouTube individually.")

    csv_path = dest / "musiccaps-public.csv"
    if not csv_path.exists():
        download(MUSICCAPS_CSV, csv_path)
    else:
        print(f"  have {csv_path.name}")

    if not _have("yt-dlp"):
        print("\n  yt-dlp is not installed -- captions downloaded, audio skipped.")
        print("    pip install yt-dlp")
        print("  ffmpeg is also required to cut the 10 s window.")
        return
    if not _have("ffmpeg"):
        print("\n  WARNING: ffmpeg not found. yt-dlp cannot extract/cut audio without")
        print("  it, so every clip will fail. Install ffmpeg and re-run.")

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = [r for r in _csv.DictReader(fh) if r.get("ytid")]
    print(f"  {len(rows)} caption rows")

    # Stratify by the clip's first AudioSet label so a subset still spans
    # genres; prefer the balanced subset the dataset authors marked.
    def stratum(row: dict) -> str:
        labels = (row.get("audioset_positive_labels") or "").split(",")
        return labels[0].strip() or "unknown"

    rng = _random.Random(seed)
    buckets: dict[str, list] = {}
    for row in sorted(rows, key=lambda r: (r.get("is_balanced_subset") != "True",)):
        buckets.setdefault(stratum(row), []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked, pools = [], [b for _, b in sorted(buckets.items())]
    while len(picked) < min(n_clips, len(rows)) and any(pools):
        for pool in pools:
            if pool and len(picked) < n_clips:
                picked.append(pool.pop())
    print(f"  targeting {len(picked)} clips across {len(buckets)} AudioSet strata")

    manifest_path = dest / "download_manifest.json"
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else {}

    ok = sum(1 for v in manifest.values() if v == "ok")
    failed = attempted = 0
    try:
        for i, row in enumerate(picked, 1):
            ytid = row["ytid"]
            target = audio_dir / f"{ytid}.wav"
            if target.exists():
                manifest[ytid] = "ok"
                continue
            if manifest.get(ytid) == "gone":     # permanent failure: do not retry
                continue

            start = float(row.get("start_s") or 0)
            end = float(row.get("end_s") or start + 10)
            attempted += 1
            cmd = [
                "yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
                "--download-sections", f"*{start}-{end}",
                "--force-keyframes-at-cuts", "--no-playlist",
                "--retries", "2", "--socket-timeout", "20",
                "--quiet", "--no-warnings", "--no-progress",
                "-o", str(audio_dir / "%(id)s.%(ext)s"),
                f"https://www.youtube.com/watch?v={ytid}",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            except subprocess.TimeoutExpired:
                manifest[ytid] = "timeout"
                failed += 1
            else:
                if target.exists():
                    manifest[ytid] = "ok"
                    ok += 1
                else:
                    err = (proc.stderr or "").lower()
                    permanent = any(s in err for s in (
                        "video unavailable", "private video", "removed",
                        "terminated", "does not exist", "age-restricted"))
                    manifest[ytid] = "gone" if permanent else "error"
                    failed += 1

            if i % 25 == 0 or i == len(picked):
                manifest_path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
                rate = ok / max(ok + failed, 1)
                print(f"  [{i}/{len(picked)}] have {ok} clips, {failed} failed "
                      f"({rate:.0%} success)")
                if attempted >= 40 and rate < 0.15:
                    print("\n  Success rate is very low -- YouTube is most likely "
                          "rate-limiting or\n  blocking this host. Stopping early. "
                          "Options: retry later, use a\n  different network, or run "
                          "Task 4 on MagnaTagATune and document it.")
                    break
    except KeyboardInterrupt:
        print("\n  interrupted -- manifest saved, re-run to resume")

    manifest_path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
    have = len(list(audio_dir.glob("*.wav")))
    print(f"\n  {have} clips on disk -> {audio_dir}")
    print(f"  manifest: {manifest_path.name}")
    if have < 200:
        print("\n  Fewer than 200 clips: too few for a meaningful retrieval split.")
        print("  Re-run to resume, or fall back to MagnaTagATune for Task 4.")
    else:
        print("\n  Next:")
        print("    python src/preprocess.py --set dataset.source=musiccaps \\")
        print("        --set paths.processed=data/processed_musiccaps \\")
        print("        --set paths.splits=data/splits_musiccaps")


def fetch(name: str, do_extract: bool, keep_zip: bool, verify: str | None) -> None:
    spec = DATASETS[name]
    dest = RAW / spec["dest"]
    archive = dest / Path(spec["url"]).name

    print(f"\n== {name} ==  ~{spec['approx_gb']} GB")
    print(f"  {spec['note']}")

    have = archive.stat().st_size / 1073741824 if archive.exists() else 0.0
    needed = spec["approx_gb"] - have + (spec["approx_gb"] if do_extract else 0)
    free = disk_free_gb(RAW)
    print(f"  disk free: {free:.1f} GB   needed: ~{needed:.1f} GB")
    if free < needed:
        print(f"  WARNING: not enough free space. Free some, or extract and "
              f"delete archives one at a time.")

    download(spec["url"], archive)

    if verify:
        print("  hashing (this takes a minute on a multi-GB file)...")
        got = sha1(archive)
        if got.lower() != verify.lower():
            raise SystemExit(f"  SHA1 MISMATCH\n    expected {verify}\n    got      {got}")
        print(f"  sha1 OK: {got}")

    if do_extract:
        extract(archive, dest, delete_after=not keep_zip)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download the datasets in the spec's Table 1, resumably.")
    ap.add_argument("dataset", nargs="*",
                    help=f"one or more of: {', '.join(DATASETS)}, magnatagatune, "
                         f"musiccaps, or a group: {', '.join(ALIASES)}")
    ap.add_argument("--clips", type=int, default=1200,
                    help="musiccaps only: how many YouTube clips to attempt")
    ap.add_argument("--extract", action="store_true", help="unzip after downloading")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep the archive after extracting (default: delete, to save disk)")
    ap.add_argument("--sha1", default=None,
                    help="expected SHA1 (copy it from the dataset's own README)")
    ap.add_argument("--list", action="store_true", help="show datasets and exit")
    args = ap.parse_args()

    if args.list or not args.dataset:
        print("Available datasets:\n")
        for name, spec in DATASETS.items():
            print(f"  {name:20s} ~{spec['approx_gb']:6.2f} GB   {spec['note']}")
        # Both of these need bespoke handlers, so they are not in DATASETS.
        print(f"  {'magnatagatune':20s} ~  2.80 GB   "
              f"25,877 clips / 29 s, 188 tags. 3-part split zip.")
        print(f"  {'musiccaps':20s} ~  0.30 GB   "
              f"5,521 clips / 10 s + expert captions. Audio via yt-dlp.")
        print("\nGroups:")
        for alias, members in ALIASES.items():
            print(f"  {alias:20s} = {', '.join(members)}")
        print("\nExample:  python tools/fetch_data.py fma_small --extract")
        return 0

    # MusicCaps pulls its audio from YouTube per-clip; its own handler.
    if "musiccaps" in args.dataset:
        fetch_musiccaps(args.clips)
        args.dataset = [d for d in args.dataset if d != "musiccaps"]
        if not args.dataset:
            return 0

    # MagnaTagATune has its own multi-part handler.
    if "magnatagatune" in args.dataset or "mtat" in args.dataset:
        fetch_magnatagatune(args.extract, args.keep_zip)
        args.dataset = [d for d in args.dataset if d not in ("magnatagatune", "mtat")]
        if not args.dataset:
            print("\ndone. Next:")
            print("  python src/preprocess.py --set dataset.source=magnatagatune")
            return 0

    names: list[str] = []
    for item in args.dataset:
        if item in ALIASES:
            names.extend(ALIASES[item])
        elif item in DATASETS:
            names.append(item)
        else:
            raise SystemExit(f"unknown dataset {item!r}; --list shows the options")

    if args.sha1 and len(names) > 1:
        raise SystemExit("--sha1 applies to a single dataset")

    for name in names:
        fetch(name, args.extract, args.keep_zip, args.sha1)

    print("\ndone. Next:")
    print("  python src/preprocess.py --set dataset.source=fma_small")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
