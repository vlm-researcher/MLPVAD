import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, cross_attention=False, kv_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads."

        kv_dim = kv_dim if kv_dim is not None else dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(kv_dim, dim)
        self.v_proj = nn.Linear(kv_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.cross_attn=cross_attention

    def forward(self, query, key, value, attention_mask=None, return_attention=True):
        B, Nq, C = query.shape
        B, Nk, _ = key.shape

        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(1, 2).reshape(B, Nq, C)
        output = self.out_proj(attn_output)

        if return_attention and self.cross_attn:
            return output, attn_probs  # <<<<<< return attention maps!
        else:
            return output

class FeedForward(nn.Module):
    def __init__(self, dim, intermediate_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class QFormerLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, cross_attention=False, kv_dim=None):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(dim, num_heads, dropout, cross_attention=True, kv_dim=kv_dim) if cross_attention else None
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if cross_attention else None
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, query, encoder_hidden_states=None, attention_mask=None, return_cross_attention=True):
        cross_attn_map = None
        query = query + self.self_attn(self.norm1(query), self.norm1(query), self.norm1(query))
        normed_query = self.norm1(query)
        
        if self.cross_attention is not None and encoder_hidden_states is not None:
            normed_kv = self.norm2(encoder_hidden_states)
            cross_attn_output, attn_weights = self.cross_attention(
                normed_query, normed_kv, normed_kv, attention_mask, return_attention=True)
            
            query = query + cross_attn_output

        query = query + self.mlp(self.norm3(query))
        
        if return_cross_attention:
            cross_attn_map = attn_weights  # shape: (B, heads, queries, frames)

        return query, cross_attn_map

class QFormer(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=6, num_heads=8, mlp_ratio=4.0, dropout=0.1, kv_dim=512):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i % 2 == 0:
                # Even layers: with cross-attention
                self.layers.append(QFormerLayer(hidden_dim, num_heads, mlp_ratio, dropout, cross_attention=True, kv_dim=kv_dim))
            else:
                # Odd layers: self-attention only
                self.layers.append(QFormerLayer(hidden_dim, num_heads, mlp_ratio, dropout, cross_attention=True))

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_embeds, encoder_hidden_states, attention_mask=None, return_attention=True):
        cross_attn_maps = []  # collect attention maps
        for layer in self.layers:
            query_embeds, cross_attn_map = layer(query_embeds, encoder_hidden_states, attention_mask)

            if return_attention and cross_attn_map is not None:
                cross_attn_maps.append(cross_attn_map)

        if return_attention:
            # Stack across layers if needed
            return query_embeds, cross_attn_maps[-1]
        else:
            return query_embeds, _

# if __name__ == "__main__":
#     qformer = QFormer(num_queries=32, hidden_dim=768, num_layers=12, num_heads=12)
#     print(qformer)
#     dummy_encoder_features = torch.randn(4, 257, 1408)
    
#     output = qformer(dummy_encoder_features)
#     print(output.shape)  # Expected: (4, 32, 768) output