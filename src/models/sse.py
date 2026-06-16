import torch
import torch.nn as nn


class SinusoidalSpatialEmbedding(nn.Module):
    """
    输入 bbox_norm_xywh: [B, 4]
    输出高频空间嵌入: [B, 4 * num_freqs * 2]
    """
    def __init__(self, num_freqs=16):
        super().__init__()
        self.num_freqs = num_freqs

        freq_bands = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freq_bands", freq_bands)

    @property
    def out_dim(self):
        return 4 * self.num_freqs * 2

    def forward(self, x):
        # x: [B, 4]
        # x[..., None] -> [B, 4, 1]
        xb = x.unsqueeze(-1) * self.freq_bands  # [B, 4, F]
        emb = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)  # [B, 4, 2F]
        emb = emb.reshape(x.shape[0], -1)
        return emb