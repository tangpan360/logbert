import torch
import torch.nn as nn
from .multi_head import MultiHeadedAttention


class MultiBranch(nn.Module):
    def __init__(self, h, d_model):
        super(MultiBranch, self).__init__()
        self.multi_attention = MultiHeadedAttention(h=h, d_model=d_model)
        self.conv = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, padding_mode='circular', bias=False)

    def forward(self, query, key, value, pos_embed, mask=None):
        # global k, v
        out = []
        attn = None
        start = 0
        embed_dim = 128

        q = query[..., start:start+embed_dim]
        k = key[..., start:start+embed_dim]
        v = value[..., start:start+embed_dim]

        # if key is not None:
        #     assert value is not None
        #     k, v = key[..., start: start+embed_dim], value[..., start: start+embed_dim]
        #     start += embed_dim
        # k, v = key[..., start: start + embed_dim], value[..., start: start + embed_dim]
        start += embed_dim

        x1 = self.multi_attention(q, k, v, pos_embed, mask=mask)
        out.append(x1)

        q = query[..., start:start+embed_dim]

        x2 = self.conv(q.permute(0, 2, 1)).transpose(1, 2)

        out.append(x2)

        out = torch.cat(out, dim=-1)
        return out

