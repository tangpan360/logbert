import joblib
import pickle
import torch.nn as nn
import torch

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .embedding.token import TokenEmbedding
from .embedding.position import PositionalEmbedding

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

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=max_len)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 2, dropout) for _ in range(n_layers)])


    def forward(self, x, segment_info=None, time_info=None, i=None, totol_length=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        global attns
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x_position = self.position(x)
        x, x_original = self.embedding(x, segment_info, time_info)
        x_position = x_position.expand(x_original.shape)

        if i != totol_length-1:
        # if i != 0:
            # running over multiple transformer blocks
            for transformer in self.transformer_blocks:
                x, attn, attn_ww, attn_wp, attn_pw, attn_pp = transformer.forward(x, mask, x_original, x_position)
        else:
            # running over multiple transformer blocks
            attns = []
            attns_ww = []
            attns_wp = []
            attns_pw = []
            attns_pp = []
            for transformer in self.transformer_blocks:
                x, attn, attn_ww, attn_wp, attn_pw, attn_pp = transformer.forward(x, mask, x_original, x_position)
                attns.append(attn)
                attns_ww.append(attn_ww)
                attns_wp.append(attn_wp)
                attns_pw.append(attn_pw)
                attns_pp.append(attn_pp)
            attns_cpu = [attn.cpu() for attn in attns]
            attns_ww_cpu = [attn.cpu() for attn in attns_ww]
            attns_wp_cpu = [attn.cpu() for attn in attns_wp]
            attns_pw_cpu = [attn.cpu() for attn in attns_pw]
            attns_pp_cpu = [attn.cpu() for attn in attns_pp]
            joblib.dump(attns_cpu, '../output/tbird/attns.pkl')
            joblib.dump(attns_ww_cpu, '../output/tbird/attns_ww.pkl')
            joblib.dump(attns_wp_cpu, '../output/tbird/attns_wp.pkl')
            joblib.dump(attns_pw_cpu, '../output/tbird/attns_pw.pkl')
            joblib.dump(attns_pp_cpu, '../output/tbird/attns_pp.pkl')
            # with open('/home/iip/tp/logbert/output/tbird/attns.pkl', 'wb') as f:
            #     pickle.dump(attns_cpu, f)

        return x
