# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        models:
            attention.py
"""
# ============================ Third Party libs ============================
from typing import Optional, Tuple
import numpy as np
import torch


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled Dot-Product Attention
    Inputs: query, key, value, mask
        - query (batch, q_len, d_model): tensor containing projection vector for decoder.
        - key (batch, k_len, d_model): tensor containing projection vector for encoder.
        - value (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - mask (-): tensor containing indices to be masked
    """

    def __init__(self, dim: int):
        super().__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.wq = torch.nn.Linear(in_features=dim, out_features=dim)
        self.wk = torch.nn.Linear(in_features=dim, out_features=dim)
        self.wv = torch.nn.Linear(in_features=dim, out_features=dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, query.size(1), -1)
            score.masked_fill_(mask, -float("Inf"))

        attn = torch.nn.functional.softmax(score, dim=-1)

        context = torch.bmm(attn, value)
        return context, attn
