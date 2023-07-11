import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, pos_attn, mask=None, dropout=None):
        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))
        tok_attn = torch.matmul(query, key.transpose(-2, -1))
        scores = (tok_attn + pos_attn) / math.sqrt(2 * query.size(-1))
        # scale = math.sqrt(2 * query.size(-1))
        # print('scale:', scale)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
