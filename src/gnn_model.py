"""GNN branch: GraphSAGE / GAT over music structure graphs (spec section 4.2).

GraphSAGE update implemented by `SAGEConv`:

    h_i^(l+1) = sigma( W^(l) . CONCAT( h_i^(l), MEAN_{j in N(i)} h_j^(l) ) )

Readout is mean pooling over nodes (optionally mean+max, which keeps a little
more of the "is there a loud segment anywhere" signal that mean washes out):

    g = (1/|V|) sum_i h_i^(L)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_add_pool, global_max_pool, global_mean_pool


class GNNEncoder(nn.Module):
    """Stack of message-passing layers + graph readout -> graph vector `g`."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3,
                 conv: str = "sage", dropout: float = 0.2, heads: int = 4,
                 readout: str = "mean"):
        super().__init__()
        if num_layers < 1:
            raise ValueError("gnn.num_layers must be >= 1")
        self.conv_kind = conv
        self.readout_kind = readout
        self.dropout = dropout

        # Raw node features are a 300+ dim mel/chroma/MFCC concat on wildly
        # different scales; a LayerNorm up front keeps early training stable.
        self.input_norm = nn.LayerNorm(in_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim = in_dim
        for _ in range(num_layers):
            if conv == "sage":
                layer = SAGEConv(dim, hidden_dim, aggr="mean")
                out_dim = hidden_dim
            elif conv == "gat":
                # concat=True -> heads * hidden_dim; keep the block width at
                # hidden_dim so `sage` and `gat` stay parameter-comparable.
                per_head = max(hidden_dim // heads, 1)
                layer = GATConv(dim, per_head, heads=heads, concat=True,
                                dropout=dropout)
                out_dim = per_head * heads
            else:
                raise ValueError(f"unknown gnn.conv={conv!r}; expected sage|gat")
            self.convs.append(layer)
            self.norms.append(nn.LayerNorm(out_dim))
            dim = out_dim

        self.hidden_dim = dim
        self.out_dim = dim * (2 if readout == "mean+max" else 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor | None = None, return_nodes: bool = False):
        h = self.input_norm(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = norm(conv(h, edge_index))
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Residual once the width is stable (all layers after the first).
            h = h_new + h if h.shape == h_new.shape else h_new

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        if self.readout_kind == "mean":
            g = global_mean_pool(h, batch)
        elif self.readout_kind == "max":
            g = global_max_pool(h, batch)
        elif self.readout_kind == "sum":
            g = global_add_pool(h, batch)
        elif self.readout_kind == "mean+max":
            g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        else:
            raise ValueError(f"unknown gnn.readout={self.readout_kind!r}")

        return (g, h) if return_nodes else g


class GNNTagClassifier(nn.Module):
    """Task 2: genre / tag prediction from the graph alone (audio-only features).

        g = READOUT(GNN(G)),   y_hat = sigmoid(W g + b)
    """

    def __init__(self, in_dim: int, num_tags: int, hidden_dim: int = 128,
                 num_layers: int = 3, conv: str = "sage", dropout: float = 0.2,
                 heads: int = 4, readout: str = "mean"):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, num_layers, conv,
                                  dropout, heads, readout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, num_tags),
        )
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, data, return_embedding: bool = False):
        g = self.encoder(data.x, data.edge_index, getattr(data, "batch", None))
        logits = self.head(g)
        return (logits, g) if return_embedding else logits


@torch.no_grad()
def graph_coherence_score(encoder: GNNEncoder, data, threshold: float = 0.5,
                          detailed: bool = False):
    """Spec section 6: do linked nodes agree after encoding?

        S_graph = (1/|E|) sum_{(i,j) in E} 1[ cos(h_i, h_j) > tau ]

    Reported on its own this number is close to meaningless. Node features come
    out of a ReLU, so they live in the non-negative orthant where cosine
    similarity between *any* two vectors is already high -- measured here,
    S_graph saturates at ~0.999 for every graph and every network depth, edges
    and non-edges alike.

    So the useful quantity is the **lift** over unlinked pairs within the same
    graph: how much more coherent a real edge is than a random non-edge. A lift
    near zero means the encoding carries no structural information, whatever
    S_graph says. `detailed=True` returns the components.
    """
    encoder.eval()
    _, h = encoder(data.x, data.edge_index, getattr(data, "batch", None),
                   return_nodes=True)

    src, dst = data.edge_index
    keep = src != dst                       # self-loops are trivially coherent
    if keep.sum() == 0:
        return {"s_graph": float("nan")} if detailed else float("nan")
    edge_sim = F.cosine_similarity(h[src[keep]], h[dst[keep]], dim=-1)
    s_graph = float((edge_sim > threshold).float().mean())

    # Sample non-edges from the same graph(s) as the contrast set.
    n = h.size(0)
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=h.device)
    linked = set(zip(src.tolist(), dst.tolist()))
    generator = torch.Generator(device="cpu").manual_seed(0)
    n_samples = int(keep.sum())
    a = torch.randint(0, n, (n_samples * 3,), generator=generator).to(h.device)
    b = torch.randint(0, n, (n_samples * 3,), generator=generator).to(h.device)
    valid = (a != b) & (batch[a] == batch[b])
    pairs = [(i, j) for i, j in zip(a[valid].tolist(), b[valid].tolist())
             if (i, j) not in linked][:n_samples]

    if not pairs:
        return ({"s_graph": s_graph, "non_edge_similarity": float("nan"),
                 "coherence_lift": float("nan")} if detailed else s_graph)

    pa = torch.tensor([i for i, _ in pairs], device=h.device)
    pb = torch.tensor([j for _, j in pairs], device=h.device)
    non_edge_sim = F.cosine_similarity(h[pa], h[pb], dim=-1)

    if not detailed:
        return s_graph
    return {
        "s_graph": s_graph,
        "edge_similarity": float(edge_sim.mean()),
        "non_edge_similarity": float(non_edge_sim.mean()),
        "coherence_lift": float(edge_sim.mean() - non_edge_sim.mean()),
    }


def build_gnn(cfg, in_dim: int, num_tags: int) -> GNNTagClassifier:
    g = cfg["gnn"]
    return GNNTagClassifier(
        in_dim=in_dim,
        num_tags=num_tags,
        hidden_dim=g["hidden_dim"],
        num_layers=g["num_layers"],
        conv=g["conv"],
        dropout=g["dropout"],
        heads=g.get("heads", 4),
        readout=g.get("readout", "mean"),
    )
