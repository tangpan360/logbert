import torch
import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        # self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.linear_layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(3)])
        # self.output_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(128, 128)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, pos_embed, mask=None):
        batch_size = query.size(0)

        pos_embed1 = pos_embed[..., 0:128]

        pos_query, pos_key = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                              for l, x in zip(self.linear_layers, (pos_embed1, pos_embed1))]
        pos_attn = torch.matmul(pos_query, pos_key.transpose(-2, -1))
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, pos_attn, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
