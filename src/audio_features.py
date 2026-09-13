"""Audio front-end: log-mel / chroma / MFCC extraction and segmentation.

Pipeline (spec section 3):
    1. resample to 22,050 Hz
    2. log-mel (128 bins) or chroma (12 bins), per-track normalised
    3. split into fixed 5-10 s windows, or beat-synchronous segments

librosa is the reference implementation. A small numpy STFT fallback keeps the
pipeline runnable when librosa is unavailable, so graph construction and model
code can be exercised without the audio stack installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # pragma: no cover - depends on which backend is installed
    import librosa

    HAVE_LIBROSA = True
except ImportError:  # pragma: no cover
    librosa = None
    HAVE_LIBROSA = False


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_audio(path, sample_rate: int = 22050, duration: float | None = None) -> np.ndarray:
    """Load a mono waveform resampled to `sample_rate`."""
    if HAVE_LIBROSA:
        y, _ = librosa.load(str(path), sr=sample_rate, mono=True, duration=duration)
        return y.astype(np.float32)

    import wave

    with wave.open(str(path), "rb") as wf:  # fallback: 16-bit PCM WAV only
        frames = wf.readframes(wf.getnframes())
        n_ch = wf.getnchannels()
        sr_in = wf.getframerate()
    y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        y = y.reshape(-1, n_ch).mean(axis=1)
    if sr_in != sample_rate:  # linear resample is adequate for the fallback
        n_out = int(round(len(y) * sample_rate / sr_in))
        y = np.interp(np.linspace(0, len(y) - 1, n_out), np.arange(len(y)), y)
    if duration is not None:
        y = y[: int(duration * sample_rate)]
    return y.astype(np.float32)


# --------------------------------------------------------------------------- #
# Spectral features
# --------------------------------------------------------------------------- #
def _stft_power(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Magnitude-squared STFT with a Hann window (numpy fallback)."""
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    pad = n_fft // 2
    y = np.pad(y, pad, mode="reflect")
    n_frames = 1 + (len(y) - n_fft) // hop_length
    if n_frames < 1:
        y = np.pad(y, (0, n_fft - len(y)))
        n_frames = 1
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = y[idx] * window
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return (np.abs(spec) ** 2).T.astype(np.float32)  # (freq, time)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    hz_pts = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2))
    fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)
    enorm = 2.0 / np.maximum(hz_pts[2:] - hz_pts[:-2], 1e-9)
    return fb * enorm[:, None]


def log_mel_spectrogram(y: np.ndarray, sr: int = 22050, n_fft: int = 2048,
                        hop_length: int = 512, n_mels: int = 128) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, T)."""
    if HAVE_LIBROSA:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    power = _stft_power(y, n_fft, hop_length)
    mel = _mel_filterbank(sr, n_fft, n_mels) @ power
    ref = np.maximum(mel.max(), 1e-10)
    return (10.0 * np.log10(np.maximum(mel, 1e-10)) - 10.0 * np.log10(ref)).astype(np.float32)


def chromagram(y: np.ndarray, sr: int = 22050, n_fft: int = 2048,
               hop_length: int = 512, n_chroma: int = 12) -> np.ndarray:
    """Chroma (pitch-class) features, shape (n_chroma, T). Drives chord graphs."""
    if HAVE_LIBROSA:
        return librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_chroma=n_chroma).astype(np.float32)
    power = _stft_power(y, n_fft, hop_length)
    freqs = np.linspace(0, sr / 2, power.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-9) / 440.0)
    pitch_class = np.mod(np.round(np.nan_to_num(midi)), 12).astype(int)
    valid = np.isfinite(midi) & (freqs > 20.0)
    chroma = np.zeros((n_chroma, power.shape[1]), dtype=np.float32)
    for pc in range(n_chroma):
        mask = valid & (pitch_class == pc)
        if mask.any():
            chroma[pc] = power[mask].sum(axis=0)
    return (chroma / np.maximum(chroma.max(axis=0, keepdims=True), 1e-9)).astype(np.float32)


def mfcc(y: np.ndarray, sr: int = 22050, n_fft: int = 2048,
         hop_length: int = 512, n_mfcc: int = 20) -> np.ndarray:
    """MFCCs, shape (n_mfcc, T). Used for segment-similarity edges."""
    if HAVE_LIBROSA:
        return librosa.feature.mfcc(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc).astype(np.float32)
    logmel = log_mel_spectrogram(y, sr, n_fft, hop_length, n_mels=40)
    n_mels = logmel.shape[0]
    basis = np.cos(np.pi / n_mels
                   * np.outer(np.arange(n_mfcc), np.arange(n_mels) + 0.5)).astype(np.float32)
    basis[0] *= np.sqrt(0.5)
    return (np.sqrt(2.0 / n_mels) * (basis @ logmel)).astype(np.float32)


def normalize_per_track(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean / unit-variance per feature bin across the whole track."""
    mu = feat.mean(axis=1, keepdims=True)
    sigma = feat.std(axis=1, keepdims=True)
    return ((feat - mu) / (sigma + eps)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def fixed_window_bounds(n_frames: int, sr: int, hop_length: int,
                        segment_seconds: float) -> list:
    """Frame index ranges for non-overlapping fixed windows."""
    frames_per_seg = max(int(round(segment_seconds * sr / hop_length)), 1)
    bounds = [(s, min(s + frames_per_seg, n_frames))
              for s in range(0, n_frames, frames_per_seg)]
    # Fold a runt tail (< half a window) into its predecessor.
    if len(bounds) > 1 and (bounds[-1][1] - bounds[-1][0]) < frames_per_seg // 2:
        bounds.pop()
        bounds[-1] = (bounds[-1][0], n_frames)
    return bounds


def beat_sync_bounds(y: np.ndarray, sr: int, hop_length: int, n_frames: int) -> list:
    """Beat-synchronous segment bounds (librosa beat tracker)."""
    if not HAVE_LIBROSA:
        raise RuntimeError("beat_sync segmentation requires librosa")
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="frames")
    edges = np.unique(np.concatenate([[0], np.asarray(beats, dtype=int), [n_frames]]))
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return bounds or [(0, n_frames)]


@dataclass
class TrackFeatures:
    """Everything the graph builder needs from one track."""

    track_id: str
    node_features: np.ndarray              # (n_segments, feature_dim)
    chroma_segments: np.ndarray            # (n_segments, n_chroma) -- chord graphs
    mfcc_segments: np.ndarray              # (n_segments, 2*n_mfcc) -- similarity edges
    bounds: list = field(default_factory=list)
    sample_rate: int = 22050
    hop_length: int = 512

    @property
    def n_segments(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.node_features.shape[1])

    def segment_times(self) -> list:
        scale = self.hop_length / self.sample_rate
        return [(round(a * scale, 3), round(b * scale, 3)) for a, b in self.bounds]


def _pool(feat: np.ndarray, bounds: list) -> np.ndarray:
    """Mean+std pool a (dim, T) feature matrix over segment bounds."""
    rows = []
    for a, b in bounds:
        chunk = feat[:, a:b]
        if chunk.shape[1] == 0:
            chunk = feat[:, max(a - 1, 0): max(a, 1)]
        rows.append(np.concatenate([chunk.mean(axis=1), chunk.std(axis=1)]))
    return np.stack(rows).astype(np.float32)


def extract_track_features(y: np.ndarray, track_id: str, cfg) -> TrackFeatures:
    """Waveform -> per-segment node features (spec preprocessing steps 1-3)."""
    a = cfg["audio"]
    sr, hop = a["sample_rate"], a["hop_length"]
    logmel = normalize_per_track(log_mel_spectrogram(y, sr, a["n_fft"], hop, a["n_mels"]))
    chroma = chromagram(y, sr, a["n_fft"], hop, a["n_chroma"])
    mfccs = normalize_per_track(mfcc(y, sr, a["n_fft"], hop, a["n_mfcc"]))

    n_frames = logmel.shape[1]
    if a.get("beat_sync", False) and HAVE_LIBROSA:
        bounds = beat_sync_bounds(y, sr, hop, n_frames)
    else:
        bounds = fixed_window_bounds(n_frames, sr, hop, a["segment_seconds"])

    mel_seg = _pool(logmel, bounds)                          # (n_seg, 2*n_mels)
    chroma_seg = _pool(chroma, bounds)[:, : a["n_chroma"]]   # means only
    mfcc_seg = _pool(mfccs, bounds)                          # (n_seg, 2*n_mfcc)

    node_features = np.concatenate([mel_seg, chroma_seg, mfcc_seg], axis=1)
    return TrackFeatures(
        track_id=track_id,
        node_features=node_features.astype(np.float32),
        chroma_segments=chroma_seg.astype(np.float32),
        mfcc_segments=mfcc_seg.astype(np.float32),
        bounds=bounds,
        sample_rate=sr,
        hop_length=hop,
    )


def node_feature_dim(cfg) -> int:
    """Dimensionality produced by `extract_track_features` (for model wiring)."""
    a = cfg["audio"]
    return 2 * a["n_mels"] + a["n_chroma"] + 2 * a["n_mfcc"]
