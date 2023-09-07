import joblib
import pickle
import torch.nn as nn
import torch

from .transformer import TransformerBlock
from .embedding import BERTEmbedding

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, max_len=512, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, is_logkey=True, is_time=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, max_len=max_len, is_logkey=is_logkey, is_time=is_time)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 2, dropout) for _ in range(n_layers)])


    def forward(self, x, segment_info=None, time_info=None, i=None, totol_length=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        global attns
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info, time_info)

        if i != totol_length-1:
        # if i != 0:
            # running over multiple transformer blocks
            for transformer in self.transformer_blocks:
                x, attn = transformer.forward(x, mask)
        else:
            # running over multiple transformer blocks
            attns = []
            for transformer in self.transformer_blocks:
                x, attn = transformer.forward(x, mask)
                attns.append(attn)
            attns_cpu = [attn.cpu() for attn in attns]
            joblib.dump(attns_cpu, '../output/tbird/attns.pkl')
            # with open('/home/iip/tp/logbert/output/tbird/attns.pkl', 'wb') as f:
            #     pickle.dump(attns_cpu, f)

        return x
