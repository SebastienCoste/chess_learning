import torch.nn as nn
import torch
import torch.nn.functional as F

class ChessMultiStageAttention(nn.Module):
    """5-stage self-attention specifically designed for chess patterns"""

    def __init__(self, embed_dims=[19, 64, 96, 96, 96], num_heads=[1, 8, 8, 8, 4]):
        super(ChessMultiStageAttention, self).__init__()

        self.attention_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.position_embeddings = nn.ParameterList()

        print("Attention configuration:")
        for i, (embed_dim, heads) in enumerate(zip(embed_dims, num_heads)):
            assert embed_dim % heads == 0, f"Stage {i}: {embed_dim} % {heads} != 0"
            print(f"Stage {i}: embed_dim={embed_dim}, heads={heads}, head_dim={embed_dim // heads}")
            # Chess-specific attention with positional bias
            attention = ChessSpecificAttention(
                embed_dim=embed_dim,
                num_heads=heads,
                spatial_size=8 if i < 2 else 4,  # Higher res for early layers
                chess_bias=True
            )
            self.attention_layers.append(attention)
            self.layer_norms.append(nn.LayerNorm(embed_dim))

            # Learnable positional embeddings for each stage
            pos_size = 8 if i < 2 else 4
            # Create parameter directly (no nn.Parameter wrapper needed)
            pos_embed = torch.randn(1, embed_dim, pos_size, pos_size)
            # FIX: Append as ParameterList element
            self.position_embeddings.append(nn.Parameter(pos_embed))
            nn.init.xavier_uniform_(self.position_embeddings[i])

    def forward(self, x, stage):
        """Apply attention at specific stage"""
        # Add stage-specific positional encoding
        x = x + self.position_embeddings[stage]

        # Reshape for attention
        batch, channels, height, width = x.shape
        x_flat = x.view(batch, channels, -1).transpose(1, 2)

        # Apply layer norm (pre-norm architecture)
        x_norm = self.layer_norms[stage](x_flat)

        # Self-attention with residual connection
        attn_out = self.attention_layers[stage](x_norm)
        x_out = x_flat + attn_out

        # Reshape back to spatial format
        return x_out.transpose(1, 2).view(batch, channels, height, width)


class ChessSpecificAttention(nn.Module):
    """Self-attention with chess-specific inductive biases"""

    def __init__(self, embed_dim, num_heads, spatial_size=8, chess_bias=True):
        super(ChessSpecificAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.spatial_size = spatial_size
        self.chess_bias = chess_bias

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

        # Chess-specific positional bias
        if chess_bias:
            self.register_buffer('chess_bias_matrix',
                                 self._create_chess_bias(spatial_size))

    def _create_chess_bias(self, size):
        """Create chess-specific attention bias for piece relationships"""
        bias = torch.zeros(size * size, size * size)

        for i in range(size):
            for j in range(size):
                idx1 = i * size + j
                for x in range(size):
                    for y in range(size):
                        idx2 = x * size + y

                        # Bias for piece movement patterns
                        distance = abs(i - x) + abs(j - y)  # Manhattan distance
                        diagonal = abs(i - x) == abs(j - y)  # Diagonal movement

                        if distance == 0:  # Same square
                            bias[idx1, idx2] = 0.1
                        elif distance <= 2:  # Close squares
                            bias[idx1, idx2] = 0.05
                        elif diagonal:  # Diagonal relationships
                            bias[idx1, idx2] = 0.03

        return bias

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Generate Q, K, V
        qkv = self.qkv_proj(x)
        expected_elements = batch_size * seq_len * 3 * self.num_heads * self.head_dim
        if qkv.numel() != expected_elements:
            raise RuntimeError(f"Shape mismatch: {qkv.shape} cannot reshape to [B, S, 3, H, D]")
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Add chess-specific bias
        if self.chess_bias and hasattr(self, 'chess_bias_matrix'):
            attn_scores = attn_scores + self.chess_bias_matrix.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_out = attn_weights @ v
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_out)
