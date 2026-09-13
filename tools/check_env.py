"""Verify the interpreter has everything the pipeline needs, and say which one.

Reproducibility is explicitly graded, and the failure mode is quiet: a machine
can easily have several Python installs, only one of which has torch. Running
`python src/train.py` under the wrong one raises a bare ImportError that looks
like a missing dependency rather than a wrong interpreter.

    python tools/check_env.py
"""
from __future__ import annotations

import importlib.metadata as md
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (import name, distribution name) -- they differ for several packages.
REQUIRED = [
    ("torch", "torch"),
    ("torch_geometric", "torch-geometric"),
    ("transformers", "transformers"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("yaml", "pyyaml"),
    ("tqdm", "tqdm"),
]
OPTIONAL = [("yt_dlp", "yt-dlp")]


def parse_pins(path: Path) -> dict:
    pins = {}
    if not path.exists():
        return pins
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, version = line.partition("==")
        pins[name.strip().lower()] = version.strip()
    return pins


def main() -> int:
    print(f"interpreter : {sys.executable}")
    print(f"python      : {sys.version.split()[0]}")
    print(f"platform    : {platform.system()} {platform.release()}")
    print(f"repo        : {REPO}\n")

    pins = parse_pins(REPO / "requirements.txt")
    missing, mismatched = [], []

    for import_name, dist in REQUIRED + OPTIONAL:
        optional = (import_name, dist) in OPTIONAL
        try:
            __import__(import_name)
        except ImportError:
            (print(f"  -- {dist:20s} not installed (optional)") if optional
             else missing.append(dist))
            continue
        try:
            have = md.version(dist)
        except md.PackageNotFoundError:
            have = "?"
        want = pins.get(dist.lower())
        if want and have != want and have != "?":
            mismatched.append((dist, have, want))
            print(f"  ~  {dist:20s} {have}  (pinned {want})")
        else:
            print(f"  OK {dist:20s} {have}")

    print()
    try:
        import torch

        if torch.cuda.is_available():
            print(f"device      : cuda -- {torch.cuda.get_device_name(0)}")
        else:
            print("device      : cpu (no CUDA). Training BERT-based tasks on CPU "
                  "takes hours; see notebooks/colab_run.ipynb for a GPU run.")
    except ImportError:
        pass

    if missing:
        print(f"\nMISSING: {', '.join(missing)}")
        print("This interpreter cannot run the pipeline. Install into it with:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        print("or create a clean environment:")
        print("  python -m venv .venv && .venv\\Scripts\\activate && "
              "pip install -r requirements.txt")
        return 1

    if mismatched:
        print("\nVersions differ from the pins that produced the reported results.")
        print("Usually fine, but pin-exact if you need to reproduce numbers:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")

    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
