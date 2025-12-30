import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x
    
    def forward_f32(self, x, attn_mask=None, key_padding_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_output)
        input_dtype = x.dtype
        x = self.norm1(x.to(torch.float32))

        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x).to(input_dtype)
        return x


class MaskedTransformerEncoder(nn.Module):
    def __init__(self, depth=6, embed_dim=256, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.heads = heads
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        x: [B, T, D]
        attn_mask: [T, T] or [B, T, T] or None
        key_padding_mask: [B, T] or None
        For a binary mask, a True value indicates that the corresponding position is not allowed to attend
        """
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, :, :]
            attn_mask = attn_mask.repeat(1, self.heads, 1, 1)
            attn_mask = attn_mask.flatten(0, 1)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x