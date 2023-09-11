import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None, x_original1=None, x_original2=None,
                x_position1=None, x_position2=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        # 计算单词到单词
        scores_ww = torch.matmul(x_original1, x_original2.transpose(-2, -1)) \
                 / math.sqrt(x_original1.size(-1))

        if mask is not None:
            scores_ww = scores_ww.masked_fill(mask == 0, -1e9)

        p_attn_ww = F.softmax(scores_ww, dim=-1)

        if dropout is not None:
            p_attn_ww = dropout(p_attn_ww)

        # 计算单词到位置
        scores_wp = torch.matmul(x_original1, x_position2.transpose(-2, -1)) \
                 / math.sqrt(x_original1.size(-1))

        if mask is not None:
            scores_wp = scores_wp.masked_fill(mask == 0, -1e9)

        p_attn_wp = F.softmax(scores_wp, dim=-1)

        if dropout is not None:
            p_attn_wp = dropout(p_attn_wp)

        # 计算位置到单词
        scores_pw = torch.matmul(x_position1, x_original2.transpose(-2, -1)) \
                 / math.sqrt(x_position1.size(-1))

        if mask is not None:
            scores_pw = scores_pw.masked_fill(mask == 0, -1e9)

        p_attn_pw = F.softmax(scores_pw, dim=-1)

        if dropout is not None:
            p_attn_pw = dropout(p_attn_pw)

        # 计算单词到单词
        scores_pp = torch.matmul(x_position1, x_position2.transpose(-2, -1)) \
                 / math.sqrt(x_position1.size(-1))

        if mask is not None:
            scores_pp = scores_pp.masked_fill(mask == 0, -1e9)

        p_attn_pp = F.softmax(scores_pp, dim=-1)

        if dropout is not None:
            p_attn_pp = dropout(p_attn_pp)

        return torch.matmul(p_attn, value), p_attn, p_attn_ww, p_attn_wp, p_attn_pw, p_attn_pp
