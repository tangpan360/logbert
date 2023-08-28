import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward
from .attention import MultiBranch


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        # TODO 多分支
        self.attention = MultiBranch(h=attn_heads, d_model=hidden)
        # self.attention = MultiBranch(h=4, d_model=128)
        # self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)  # original
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, pos_embed, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, pos_embed, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
