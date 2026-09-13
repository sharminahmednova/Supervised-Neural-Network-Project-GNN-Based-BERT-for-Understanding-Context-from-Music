"""Baselines for the comparison table (spec section 8).

    B1  random / majority-prior tag predictor      (no learning at all)
    B2  CNN on the mel-spectrogram                 (no graph, no text)
    B3  BERT-only                                  (Task 1 -- see train.py task1)
    B4  PCA + MLP on hand-crafted audio features   (optional)

B2 deliberately consumes the *same* node features as the GNN, laid out as a
(segments x features) image, so the only difference between B2 and Task 2 is
the relational structure. That is what makes the comparison a statement about
graphs rather than about feature engineering.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# B1: random / prior
# --------------------------------------------------------------------------- #
class PriorBaseline:
    """Predicts each tag's training-set base rate; `random` ignores the data."""

    def __init__(self, mode: str = "prior", seed: int = 42):
        if mode not in {"prior", "random", "majority"}:
            raise ValueError(f"unknown baseline mode {mode!r}")
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.prior: np.ndarray | None = None

    def fit(self, y_train: np.ndarray) -> "PriorBaseline":
        self.prior = np.asarray(y_train, dtype=np.float64).mean(axis=0)
        return self

    def predict_proba(self, n_samples: int) -> np.ndarray:
        if self.prior is None:
            raise RuntimeError("call fit() first")
        k = self.prior.shape[0]
        if self.mode == "random":
            return self.rng.random((n_samples, k))
        if self.mode == "majority":
            return np.tile((self.prior >= 0.5).astype(np.float64), (n_samples, 1))
        # `prior`: emit the base rate, jittered so ties break and AUC-PR is defined
        return np.clip(np.tile(self.prior, (n_samples, 1))
                       + self.rng.normal(0, 1e-3, (n_samples, k)), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# B2: CNN on mel-spectrogram
# --------------------------------------------------------------------------- #
class MelCNN(nn.Module):
    """Small 2-D CNN over the (segments x features) matrix of a track.

    Graphs in a batch have differing node counts, so the forward pass takes a
    dense padded tensor; `dense_batch` below builds it from a PyG batch.
    """

    def __init__(self, feature_dim: int, num_tags: int, channels=(32, 64, 128),
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_tags))
        self.input_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_segments, feature_dim)."""
        x = self.input_norm(x).unsqueeze(1)      # (B, 1, S, F)
        return self.head(self.pool(self.features(x)))


def dense_batch(batch, max_nodes: int | None = None) -> torch.Tensor:
    """PyG batch -> zero-padded dense (B, max_nodes, feature_dim) tensor."""
    from torch_geometric.utils import to_dense_batch

    x, _ = to_dense_batch(batch.x, batch.batch, max_num_nodes=max_nodes)
    return x


# --------------------------------------------------------------------------- #
# B4: PCA + MLP on hand-crafted features
# --------------------------------------------------------------------------- #
class PCAMLPBaseline:
    """Mean/std-pooled node features -> PCA -> multi-label MLP (sklearn)."""

    def __init__(self, n_components: int = 64, hidden: tuple = (128,),
                 max_iter: int = 400, seed: int = 42):
        from sklearn.decomposition import PCA
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.n_components = n_components
        self._PCA, self._MLP = PCA, MLPClassifier
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
            ("mlp", MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter,
                                  random_state=seed, early_stopping=False)),
        ])
        self.single_class_prior: np.ndarray | None = None

    @staticmethod
    def track_vector(data) -> np.ndarray:
        """One fixed-length vector per track: mean and std over its segments."""
        x = data.x.detach().cpu().numpy()
        return np.concatenate([x.mean(axis=0), x.std(axis=0)])

    @classmethod
    def featurize(cls, dataset) -> tuple:
        X = np.stack([cls.track_vector(dataset[i]) for i in range(len(dataset))])
        y = np.stack([dataset[i].y.numpy().reshape(-1) for i in range(len(dataset))])
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCAMLPBaseline":
        # PCA cannot ask for more components than samples or features.
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pipeline.set_params(pca__n_components=max(n_comp, 2))
        # sklearn's MLP drops label columns that are constant in training; keep
        # their base rate so the prediction matrix stays the full tag width.
        self.single_class_prior = y.mean(axis=0)
        self.keep = np.where((y.sum(axis=0) > 0) & (y.sum(axis=0) < y.shape[0]))[0]
        if self.keep.size == 0:
            raise RuntimeError("no tag varies in the training split")
        self.pipeline.fit(X, y[:, self.keep])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.tile(self.single_class_prior, (X.shape[0], 1))
        raw = self.pipeline.predict_proba(X)
        # MLPClassifier returns a list of per-label arrays for multi-label input
        if isinstance(raw, list):
            raw = np.stack([p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.reshape(-1)
                            for p in raw], axis=1)
        probs[:, self.keep] = raw
        return probs
