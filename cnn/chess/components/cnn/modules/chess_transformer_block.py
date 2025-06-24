import torch.nn as nn

class ChessTransformerBlock(nn.Module):
    """
    Transformer-inspired block for chess position understanding.
    Combines self-attention with feed-forward processing.
    """

    def __init__(self, embed_dim, num_heads=8, ff_dim=None, dropout=0.1):
        super(ChessTransformerBlock, self).__init__()

        if ff_dim is None:
            ff_dim = embed_dim * 4

        self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x