"""BERT text branch.

`BertEncoder` is shared by every task: it returns both the pooled CLS vector
`t` (Task 1, concat fusion, contrastive) and the full token sequence `H_text`
(cross-attention fusion, spec section 4.3).

Freezing policy is a config decision, because it dominates runtime on CPU:
    freeze_bert: true  + unfreeze_last_n: 0  -> encoder is a fixed feature map
    freeze_bert: true  + unfreeze_last_n: 2  -> top 2 transformer blocks train
    freeze_bert: false                       -> full fine-tuning (Task 1 default)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertEncoder(nn.Module):
    """Wraps a HuggingFace encoder and exposes CLS + token states."""

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 freeze: bool = True, unfreeze_last_n: int = 0):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.config.hidden_size)
        self.apply_freezing(freeze, unfreeze_last_n)

    # -- freezing ---------------------------------------------------------- #
    def _transformer_blocks(self) -> list[nn.Module]:
        """The per-layer block list, across BERT / DistilBERT / RoBERTa naming."""
        for attr in ("encoder", "transformer"):
            module = getattr(self.bert, attr, None)
            if module is None:
                continue
            for name in ("layer", "layers"):
                blocks = getattr(module, name, None)
                if blocks is not None:
                    return list(blocks)
        return []

    def apply_freezing(self, freeze: bool, unfreeze_last_n: int = 0) -> None:
        for p in self.bert.parameters():
            p.requires_grad = not freeze
        if freeze and unfreeze_last_n > 0:
            blocks = self._transformer_blocks()
            for block in blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
        self._frozen = freeze and unfreeze_last_n == 0

    # -- forward ----------------------------------------------------------- #
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Returns (cls, token_states, attention_mask).

        cls          (B, H)     -- sequence representation `t`
        token_states (B, L, H)  -- `H_text` for cross-attention
        """
        # DistilBERT's forward has no token_type_ids argument, so pass only the
        # two fields every supported encoder accepts.
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_states = out.last_hidden_state

        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = token_states[:, 0]        # DistilBERT: [CLS] is position 0
        return pooled, token_states, attention_mask

    def masked_mean(self, token_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean over real tokens -- a sturdier sentence vector than CLS when frozen."""
        mask = attention_mask.unsqueeze(-1).to(token_states.dtype)
        return (token_states * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class BertTagClassifier(nn.Module):
    """Task 1: multi-label tag classifier on text alone.

        t = BERT_CLS(X_text),   y_hat_k = sigmoid(w_k^T t + b_k)

    Returns *logits*; the trainer applies BCEWithLogitsLoss for stability.
    """

    def __init__(self, num_tags: int, model_name: str = "distilbert-base-uncased",
                 freeze: bool = False, unfreeze_last_n: int = 0,
                 dropout: float = 0.2, pooling: str = "cls"):
        super().__init__()
        self.encoder = BertEncoder(model_name, freeze, unfreeze_last_n)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.hidden_size, num_tags)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                return_attention: bool = False):
        cls, tokens, mask = self.encoder(input_ids, attention_mask)
        t = cls if self.pooling == "cls" else self.encoder.masked_mean(tokens, mask)
        logits = self.head(self.dropout(t))
        if return_attention:
            return logits, self.token_saliency(tokens, mask)
        return logits

    @torch.no_grad()
    def token_saliency(self, token_states: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """Per-token relevance: cosine of each token against the pooled vector.

        A cheap stand-in for attention rollout, enough for the qualitative
        "5 example predictions with attention visualisation" deliverable.
        """
        pooled = self.encoder.masked_mean(token_states, attention_mask)
        sim = torch.nn.functional.cosine_similarity(
            token_states, pooled.unsqueeze(1), dim=-1)
        sim = sim.masked_fill(attention_mask == 0, float("-inf"))
        return torch.softmax(sim, dim=-1)


class TextProjector(nn.Module):
    """Projects BERT states into the fusion/contrastive space."""

    def __init__(self, hidden_size: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
