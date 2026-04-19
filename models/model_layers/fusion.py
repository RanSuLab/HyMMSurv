from math import ceil

import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, reduce

class AlignFusion(nn.Module):
    #   co-attention
    def __init__(self, embedding_dim: int, num_heads: int, num_micro_genes=1):
    
        super().__init__()
    
        self.num_micro_genes = num_micro_genes
   
        self.coattn_pathology_to_microb = Attention(embedding_dim, num_heads)
        self.norm_p = nn.LayerNorm(embedding_dim)
  
        self.coattn_microb_to_pathology = Attention(embedding_dim, num_heads)
        self.norm_m = nn.LayerNorm(embedding_dim)
        
    def forward(self, token):
        """
        1.self-attention for microb
        2.cross-attention Q:microb K,V:wsi
        3.MLP for microb
        4.cross-attention Q:wsi  K,V:microb
        """
        # Align Block
        
        # Self attention block
        m = token[:,:self.num_micro_genes,:]
        p = token[:,self.num_micro_genes:,:]
        # t = token[:,g_num+p_num:,:]
        
        cross_p,attn_p = self.coattn_pathology_to_microb(k=m, q=p, v=m) # + self.coattn_patnhology_to_table(k=t, q=p, v=t)
        cross_p = self.norm_p(p + cross_p)
        attn_path = attn_p.mean(dim=1).squeeze(1)

        cross_m,attn_m = self.coattn_microb_to_pathology(k=p, q=m, v=p)
        cross_m = self.norm_m(m + cross_m)
        attn_microb = attn_m.mean(dim=1).squeeze(-1)

        output = torch.cat((cross_m, cross_p), dim=-2)

        return output,attn_path,attn_microb



    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn
