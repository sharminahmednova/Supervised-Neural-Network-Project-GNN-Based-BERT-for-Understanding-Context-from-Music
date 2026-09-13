"""Task 4: contrastive GNN-BERT dual encoder (spec section 4.4).

A shared embedding space between audio graphs and natural-language captions,
trained with InfoNCE over in-batch negatives:

    L_NCE = -log  exp(sim(g_i, t_i) / tau)
                  ---------------------------
                  sum_j exp(sim(g_i, t_j) / tau)

with sim(u, v) = u^T v / (||u|| ||v||). The loss is symmetrised over both
directions, which is what makes caption->audio and audio->caption retrieval
both work rather than only the direction that was trained.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertEncoder
from gnn_model import GNNEncoder


class ProjectionHead(nn.Module):
    """Two-layer MLP onto the shared space, L2-normalised on output."""

    def __init__(self, in_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ContrastiveGNNBert(nn.Module):
    """Dual encoder: graph -> g_hat, caption -> t_hat, both unit-norm."""

    def __init__(self, node_dim: int, cfg):
        super().__init__()
        gcfg, ccfg, tcfg = cfg["gnn"], cfg["contrastive"], cfg["text"]
        embed_dim = ccfg["embed_dim"]

        self.gnn = GNNEncoder(node_dim, gcfg["hidden_dim"], gcfg["num_layers"],
                              gcfg["conv"], gcfg["dropout"], gcfg.get("heads", 4),
                              gcfg.get("readout", "mean"))
        self.bert = BertEncoder(tcfg["model_name"], tcfg.get("freeze_bert", True),
                                tcfg.get("unfreeze_last_n", 0))
        self.graph_head = ProjectionHead(self.gnn.out_dim, embed_dim)
        self.text_head = ProjectionHead(self.bert.hidden_size, embed_dim)

        # log-space temperature keeps tau positive under unconstrained SGD
        init_log_t = float(np.log(1.0 / ccfg.get("temperature", 0.07)))
        self.logit_scale = nn.Parameter(torch.tensor(init_log_t),
                                        requires_grad=ccfg.get("learnable_temperature", True))
        self.max_logit_scale = float(np.log(100.0))

    def encode_graph(self, data) -> torch.Tensor:
        g = self.gnn(data.x, data.edge_index, getattr(data, "batch", None))
        return self.graph_head(g)

    def encode_text(self, data) -> torch.Tensor:
        cls, tokens, mask = self.bert(data.input_ids, data.attention_mask)
        # Mean pooling beats CLS when the encoder is frozen, which is the
        # default here -- a frozen CLS was never trained for this objective.
        pooled = self.bert.masked_mean(tokens, mask)
        return self.text_head(pooled)

    def forward(self, data):
        g = self.encode_graph(data)
        t = self.encode_text(data)
        scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()
        return {"graph_embed": g, "text_embed": t,
                "logits": scale * g @ t.t(), "logit_scale": scale}


def info_nce_loss(logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Symmetric InfoNCE over in-batch negatives (diagonal = the true pairs)."""
    n = logits.size(0)
    target = torch.arange(n, device=logits.device)
    loss_a2t = F.cross_entropy(logits, target)      # graph -> caption
    loss_t2a = F.cross_entropy(logits.t(), target)  # caption -> graph
    loss = 0.5 * (loss_a2t + loss_t2a)
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == target).float().mean()
    return loss, {"loss": float(loss.detach()),
                  "loss_audio2caption": float(loss_a2t.detach()),
                  "loss_caption2audio": float(loss_t2a.detach()),
                  "in_batch_acc": float(acc)}


@torch.no_grad()
def embed_split(model: ContrastiveGNNBert, loader, device) -> dict:
    """Embed a whole split once, for retrieval evaluation."""
    model.eval()
    graphs, texts, ids, captions = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        graphs.append(model.encode_graph(batch).cpu())
        texts.append(model.encode_text(batch).cpu())
        ids.extend(batch.track_id if isinstance(batch.track_id, list) else [batch.track_id])
        captions.extend(batch.text if isinstance(batch.text, list) else [batch.text])
    return {
        "graph": torch.cat(graphs).numpy(),
        "text": torch.cat(texts).numpy(),
        "track_ids": ids,
        "captions": captions,
    }


def similarity_matrix(graph_embed: np.ndarray, text_embed: np.ndarray) -> np.ndarray:
    """Cosine similarity; both inputs already unit-norm, so this is a dot product."""
    return np.asarray(graph_embed) @ np.asarray(text_embed).T


def retrieval_examples(sim: np.ndarray, captions: list[str], track_ids: list[str],
                       n: int = 10, top_k: int = 3) -> list[dict]:
    """Qualitative caption -> top-k audio results (spec deliverable: 10 examples)."""
    examples = []
    # sim[i, j] = graph i vs caption j, so caption->audio reads down a column.
    for j in range(min(n, sim.shape[1])):
        column = sim[:, j]
        order = np.argsort(-column)[:top_k]
        examples.append({
            "query_caption": captions[j],
            "ground_truth": track_ids[j],
            "rank_of_truth": int((column > column[j]).sum() + 1),
            "results": [
                {"rank": r + 1, "track_id": track_ids[i], "score": round(float(column[i]), 4),
                 "is_correct": bool(i == j), "caption": captions[i]}
                for r, i in enumerate(order)
            ],
        })
    return examples


@torch.no_grad()
def zero_shot_tag_scores(model: ContrastiveGNNBert, loader, tag_names: list[str],
                         device, prompt: str = "a music track tagged {tag}") -> tuple:
    """Zero-shot tagging: score each graph against a prompt per tag.

    Compared against the Task 3 supervised model in the spec's deliverables --
    this is the CLIP-style transfer the contrastive objective buys for free.
    """
    from transformers import AutoTokenizer

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.bert.model_name)
    prompts = [prompt.format(tag=t) for t in tag_names]
    enc = tokenizer(prompts, padding="max_length", truncation=True,
                    max_length=32, return_tensors="pt").to(device)

    cls, tokens, mask = model.bert(enc["input_ids"], enc["attention_mask"])
    tag_embed = model.text_head(model.bert.masked_mean(tokens, mask))   # (K, D)

    scores, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        g = model.encode_graph(batch)
        scores.append((g @ tag_embed.t()).cpu())
        targets.append(batch.y.float().cpu())
    return torch.cat(scores).numpy(), torch.cat(targets).numpy()
