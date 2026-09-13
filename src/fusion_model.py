"""Task 3: GNN-BERT fusion for multi-context understanding (spec section 4.3).

Cross-attention fusion, with the graph vector as the query and the BERT token
states as keys/values:

    A = softmax( Q K^T / sqrt(d) ),  Q = g W_Q,  K = H_text W_K
    z = CONCAT( g, A H_text ),       y_hat = sigmoid( W z )

Multi-task objective:

    L = L_tags + alpha ||v - v_hat||^2 + beta ||a - a_hat||^2

`mode` selects the ablation arm, so the ablation table the spec asks for
(BERT-only / GNN-only / early concat / cross-attention) is one config flag
rather than four model classes:

    cross_attention  full model
    concat           early concatenation of [g ; t], no attention
    bert_only        text branch alone (graph zeroed out)
    gnn_only         graph branch alone (text zeroed out)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertEncoder
from gnn_model import GNNEncoder

FUSION_MODES = ("cross_attention", "concat", "bert_only", "gnn_only")


class CrossAttentionFusion(nn.Module):
    """Multi-head attention from the graph vector into the text token states."""

    def __init__(self, graph_dim: int, text_dim: int, proj_dim: int,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if proj_dim % heads != 0:
            raise ValueError(f"fusion.proj_dim ({proj_dim}) must divide "
                             f"fusion.attn_heads ({heads})")
        self.heads = heads
        self.head_dim = proj_dim // heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(graph_dim, proj_dim)
        self.k_proj = nn.Linear(text_dim, proj_dim)
        self.v_proj = nn.Linear(text_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: torch.Tensor, h_text: torch.Tensor,
                attention_mask: torch.Tensor):
        """g: (B, Dg)   h_text: (B, L, Dt)   ->  (context (B, P), attn (B, H, L))."""
        b, length, _ = h_text.shape
        q = self.q_proj(g).view(b, self.heads, 1, self.head_dim)
        k = self.k_proj(h_text).view(b, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_text).view(b, length, self.heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale          # (B, H, 1, L)
        if attention_mask is not None:
            pad = (attention_mask == 0).view(b, 1, 1, length)
            scores = scores.masked_fill(pad, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)

        context = (self.dropout(attn) @ v)                       # (B, H, 1, Dh)
        context = context.transpose(1, 2).reshape(b, self.heads * self.head_dim)
        return self.norm(self.out_proj(context)), attn.squeeze(2)


class GNNBertFusion(nn.Module):
    """The Task 3 model, covering all four ablation arms."""

    def __init__(self, node_dim: int, num_tags: int, cfg, predict_emotion: bool = True):
        super().__init__()
        gcfg, fcfg, tcfg = cfg["gnn"], cfg["fusion"], cfg["text"]
        self.mode = fcfg.get("mode", "cross_attention")
        if self.mode not in FUSION_MODES:
            raise ValueError(f"unknown fusion.mode={self.mode!r}; expected {FUSION_MODES}")
        self.predict_emotion = predict_emotion
        proj_dim = fcfg["proj_dim"]
        dropout = fcfg.get("dropout", 0.3)

        self.use_graph = self.mode != "bert_only"
        self.use_text = self.mode != "gnn_only"

        if self.use_graph:
            self.gnn = GNNEncoder(node_dim, gcfg["hidden_dim"], gcfg["num_layers"],
                                  gcfg["conv"], gcfg["dropout"], gcfg.get("heads", 4),
                                  gcfg.get("readout", "mean"))
            self.graph_proj = nn.Sequential(
                nn.Linear(self.gnn.out_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        if self.use_text:
            self.bert = BertEncoder(tcfg["model_name"], tcfg.get("freeze_bert", True),
                                    tcfg.get("unfreeze_last_n", 0))
            self.text_proj = nn.Sequential(
                nn.Linear(self.bert.hidden_size, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())

        if self.mode == "cross_attention":
            self.cross = CrossAttentionFusion(
                self.gnn.out_dim, self.bert.hidden_size, proj_dim,
                fcfg.get("attn_heads", 4), dropout)
            fused_dim = proj_dim * 2          # CONCAT(g_proj, attended text)
        elif self.mode == "concat":
            fused_dim = proj_dim * 2          # CONCAT(g_proj, t_proj)
        else:
            fused_dim = proj_dim              # single branch

        self.fuse_norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Sequential(
            nn.Linear(fused_dim, proj_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, num_tags))
        nn.init.zeros_(self.tag_head[-1].bias)

        if predict_emotion:
            # Two scalars on the DEAM 1-9 scale; the trainer standardises targets.
            self.emotion_head = nn.Sequential(
                nn.Linear(fused_dim, proj_dim // 2), nn.GELU(),
                nn.Linear(proj_dim // 2, 2))

        self.fused_dim = fused_dim
        self._last_attention: torch.Tensor | None = None

    # -- forward ----------------------------------------------------------- #
    def encode(self, data):
        """Run both branches and return the fused representation `z`."""
        g = t = None
        attn = None

        if self.use_graph:
            g = self.gnn(data.x, data.edge_index, getattr(data, "batch", None))

        if self.use_text:
            ids, mask = data.input_ids, data.attention_mask
            cls, tokens, mask = self.bert(ids, mask)
            t = cls

        if self.mode == "cross_attention":
            context, attn = self.cross(g, tokens, mask)
            z = torch.cat([self.graph_proj(g), context], dim=-1)
        elif self.mode == "concat":
            z = torch.cat([self.graph_proj(g), self.text_proj(t)], dim=-1)
        elif self.mode == "bert_only":
            z = self.text_proj(t)
        else:  # gnn_only
            z = self.graph_proj(g)

        self._last_attention = attn
        return self.fuse_norm(z), attn

    def forward(self, data, return_embedding: bool = False):
        z, attn = self.encode(data)
        z = self.dropout(z)
        tag_logits = self.tag_head(z)
        emotion = self.emotion_head(z) if self.predict_emotion else None
        out = {"tag_logits": tag_logits, "emotion": emotion, "attention": attn}
        if return_embedding:
            out["z"] = z.detach()
        return out


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class MultiTaskLoss(nn.Module):
    """L = L_tags + alpha * MSE(valence) + beta * MSE(arousal).

    The emotion term is masked: corpora without DEAM annotations contribute
    only the tagging term, so a mixed corpus trains without special-casing.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, outputs: dict, data, emotion_stats=None) -> tuple:
        tag_loss = self.bce(outputs["tag_logits"], data.y.float())
        total = tag_loss
        parts = {"tag_loss": float(tag_loss.detach())}

        pred = outputs.get("emotion")
        if pred is not None and hasattr(data, "va"):
            mask = data.va_mask.reshape(-1, 1)
            if mask.sum() > 0:
                target = data.va
                if emotion_stats is not None:      # standardise for scale parity
                    mean, std = emotion_stats
                    target = (target - mean.to(target.device)) / std.to(target.device)
                sq = ((pred - target) ** 2) * mask
                denom = mask.sum().clamp(min=1.0)
                v_loss = sq[:, 0].sum() / denom
                a_loss = sq[:, 1].sum() / denom
                total = total + self.alpha * v_loss + self.beta * a_loss
                parts["valence_mse"] = float(v_loss.detach())
                parts["arousal_mse"] = float(a_loss.detach())

        parts["total"] = float(total.detach())
        return total, parts


def build_fusion_model(cfg, node_dim: int, num_tags: int,
                       predict_emotion: bool = True) -> GNNBertFusion:
    return GNNBertFusion(node_dim, num_tags, cfg, predict_emotion)


@torch.no_grad()
def top_attended_tokens(tokenizer, input_ids: torch.Tensor, attn: torch.Tensor,
                        k: int = 6) -> list[list[tuple[str, float]]]:
    """Decode the highest-attention caption tokens per example (case studies)."""
    if attn is None:
        return []
    weights = attn.mean(dim=1)                 # average the heads -> (B, L)
    out = []
    for row_ids, row_w in zip(input_ids, weights):
        scores, idx = row_w.topk(min(k, row_w.numel()))
        tokens = tokenizer.convert_ids_to_tokens(row_ids[idx].tolist())
        out.append([(tok, float(s)) for tok, s in zip(tokens, scores)])
    return out
