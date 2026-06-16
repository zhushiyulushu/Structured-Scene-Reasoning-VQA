import torch
import torch.nn as nn
from .sse import SinusoidalSpatialEmbedding


class NeuralGridInference(nn.Module):
    def __init__(self, num_rows, num_cols, num_freqs=16, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.sse = SinusoidalSpatialEmbedding(num_freqs=num_freqs)

        # raw bbox_norm_xywh 4维 + SSE高频嵌入
        in_dim = 4 + self.sse.out_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.row_head = nn.Linear(hidden_dim, num_rows)
        self.col_head = nn.Linear(hidden_dim, num_cols)

    def forward(self, bbox_norm_xywh):
        sse_emb = self.sse(bbox_norm_xywh)
        x = torch.cat([bbox_norm_xywh, sse_emb], dim=1)
        h = self.backbone(x)
        row_logits = self.row_head(h)
        col_logits = self.col_head(h)
        return row_logits, col_logits