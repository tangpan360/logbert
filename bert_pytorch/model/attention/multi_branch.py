import torch
import torch.nn as nn
from .multi_head import MultiHeadedAttention


class MultiBranch(nn.Module):
    def __init__(self, h, d_model):
        super(MultiBranch, self).__init__()
        self.multi_attention = MultiHeadedAttention(h=h, d_model=d_model)
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        # self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, padding_mode='circular', bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, query, key, value, mask=None):

        x1 = self.multi_attention(query, key, value, mask=mask)

        x2 = self.conv1(query.permute(0, 2, 1)).transpose(1, 2)
        # x2 = self.conv2(x2.permute(0, 2, 1)).transpose(1, 2)

        out = x1 + x2
        return out

