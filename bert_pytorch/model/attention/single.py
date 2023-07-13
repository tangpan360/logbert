import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self):
        super(Attention, self).__init__()
        self.bias = nn.Embedding(512, 4)
        self.num_buckets = 32
        self.max_distance = 128
        self.bidirectional = True

    def forward(self, query, key, value, mask=None, dropout=None):
        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))
        seq_len = query.size(2)
        attn = torch.matmul(query, key.transpose(-2, -1))

        relative_positions = get_relative_positions(seq_len, self.bidirectional, self.num_buckets, self.max_distance).to(attn.device)
        # relative_positions.shape == (seq_len, seq_len)
        bias = self.bias(relative_positions)
        bias = bias.permute(2, 0, 1).unsqueeze(0)

        attn = attn + bias
        scores = attn / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


def get_relative_positions(
        seq_len, bidirectional=True, num_buckets=32, max_distance=128
):
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    relative_positions = _get_relative_position_bucket(
        x - y, bidirectional, num_buckets, max_distance
    )

    return relative_positions


def _get_relative_position_bucket(
        relative_position, bidirectional, num_buckets, max_distance
):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(
            relative_position, torch.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets
