import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SpectralMoEArgs:
    dim: int = 1024
    emb_dim: int = 128
    n_layers: int = 6
    n_heads: int = 16
    n_kv_heads: int = 4
    vocab_size: int = 32768
    multiple_of: int = 256
    ffn_dim_multiplier: float = 2.0
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024
    
    # MoE
    n_experts: int = 8
    top_k: int = 2
    
    # Spectral Params
    k_dim_attn: int = 128
    k_dim_ffn: int = 128
    k_hidden_ffn: int = 256
    k_seq_len: int = 64

# =============================================================================
# 1. Utilidades Espectrales (Reutilizadas de V7)
# =============================================================================

_DCT_CACHE: dict = {}
_WALSH_ROW_CACHE: dict = {}

def _get_dct_rows(N: int, k: int) -> torch.Tensor:
    k = min(k, N)
    key = (N, k)
    if key not in _DCT_CACHE:
        i = torch.arange(k).view(k, 1)
        j = torch.arange(N).view(1, N)
        mat = torch.cos(math.pi * i * (2 * j + 1) / (2 * N))
        mat[0, :] *= 1.0 / math.sqrt(2.0)
        mat *= math.sqrt(2.0 / N)
        _DCT_CACHE[key] = mat.clone()
    return _DCT_CACHE[key]

def _get_walsh_rows(N: int, k: int) -> torch.Tensor:
    k = min(k, N)
    key = (N, k)
    if key not in _WALSH_ROW_CACHE:
        H = torch.tensor([[1.0]])
        curr_n = 1
        while curr_n < N:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
            curr_n *= 2
        _WALSH_ROW_CACHE[key] = (H[:k, :] / math.sqrt(N)).clone()
    return _WALSH_ROW_CACHE[key]

# =============================================================================
# 2. Capas Espectrales
# =============================================================================

class DCTLinear(nn.Module):
    def __init__(self, in_features, out_features, k_in, k_out, bias=False):
        super().__init__()
        self.k_in, self.k_out = min(k_in, in_features), min(k_out, out_features)
        self.in_features, self.out_features = in_features, out_features
        d_in = _get_dct_rows(in_features, self.k_in)
        d_out = _get_dct_rows(out_features, self.k_out)
        self.register_buffer('d_in_mat', d_in)
        self.register_buffer('d_out_mat', d_out)
        self.core = nn.Parameter(torch.randn(self.k_out, self.k_in) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        h = x @ self.d_in_mat.t()
        h = h @ self.core.t()
        h = h @ self.d_out_mat
        return h + self.bias if self.bias is not None else h

class WalshLinear(nn.Module):
    def __init__(self, in_features, out_features, k_in, k_out, bias=False):
        super().__init__()
        self.k_in, self.k_out = min(k_in, in_features), min(k_out, out_features)
        self.in_features, self.out_features = in_features, out_features
        w_in = 1 << (in_features-1).bit_length()
        w_out = 1 << (out_features-1).bit_length()
        self.w_in, self.w_out = w_in, w_out
        h_in = _get_walsh_rows(w_in, self.k_in)
        h_out = _get_walsh_rows(w_out, self.k_out)
        self.register_buffer('h_in_mat', h_in)
        self.register_buffer('h_out_mat', h_out)
        self.core = nn.Parameter(torch.randn(self.k_out, self.k_in) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        if x.shape[-1] != self.w_in:
            x = F.pad(x, (0, self.w_in - self.in_features))
        h = x @ self.h_in_mat.t()
        h = h @ self.core.t()
        h = h @ self.h_out_mat
        res = h[..., :self.out_features]
        return res + self.bias if self.bias is not None else res

# =============================================================================
# 3. MoE Espectral
# =============================================================================

class SpectralExpert(nn.Module):
    def __init__(self, args: SpectralMoEArgs):
        super().__init__()
        hidden = int(2 * (4 * args.dim) / 3)
        hidden = args.multiple_of * ((hidden + args.multiple_of - 1) // args.multiple_of)
        # Usamos Walsh para expertos por su velocidad superior
        self.w1 = WalshLinear(args.dim, hidden, args.k_dim_ffn, args.k_hidden_ffn)
        self.w2 = WalshLinear(hidden, args.dim, args.k_hidden_ffn, args.k_dim_ffn)
        self.w3 = WalshLinear(args.dim, hidden, args.k_dim_ffn, args.k_hidden_ffn)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SpectralMoEFeedForward(nn.Module):
    def __init__(self, args: SpectralMoEArgs):
        super().__init__()
        self.n_experts = args.n_experts
        self.top_k = args.top_k
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        self.experts = nn.ModuleList([SpectralExpert(args) for _ in range(args.n_experts)])
        
        # Inicialización del gate para evitar colapso
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        
        # Gating
        gate_logits = self.gate(x_flat)
        weights = F.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_indices, k_indices = (top_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                expert_out = expert(x_flat[token_indices])
                out[token_indices] += top_weights[token_indices, k_indices].unsqueeze(-1) * expert_out
                
        return out.view(bsz, seqlen, dim)

# =============================================================================
# 4. Bloque y Modelo Completo
# =============================================================================

class SpectralAttention(nn.Module):
    def __init__(self, args: SpectralMoEArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.k_seq_len = args.k_seq_len

        self.wq = DCTLinear(args.dim, args.n_heads * self.head_dim, args.k_dim_attn, args.k_dim_attn)
        self.wk = DCTLinear(args.dim, self.n_kv_heads * self.head_dim, args.k_dim_attn, args.k_dim_attn)
        self.wv = DCTLinear(args.dim, self.n_kv_heads * self.head_dim, args.k_dim_attn, args.k_dim_attn)
        self.wo = DCTLinear(args.n_heads * self.head_dim, args.dim, args.k_dim_attn, args.k_dim_attn)

    def forward(self, x, freqs_cis, mask, past_kv=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # Implementación simplificada de atención para el prototipo
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
        
        if self.n_rep > 1:
            xk_out = xk_out[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim).flatten(2, 3)
            xv = xv[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim).flatten(2, 3)

        # Usar SDPA nativo
        out = F.scaled_dot_product_attention(xq_out.transpose(1, 2), xk_out.transpose(1, 2), xv.transpose(1, 2), attn_mask=mask)
        return self.wo(out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)), None

class SpectralTransformerBlock(nn.Module):
    def __init__(self, args: SpectralMoEArgs):
        super().__init__()
        self.attention = SpectralAttention(args)
        self.feed_forward = SpectralMoEFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask, past_kv=None):
        h_attn, _ = self.attention(self.attention_norm(x), freqs_cis, mask, past_kv)
        h = x + h_attn
        return h + self.feed_forward(self.ffn_norm(h)), None

class SpectralThinkerMoE(nn.Module):
    def __init__(self, args: SpectralMoEArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.emb_dim)
        self.emb_proj = nn.Linear(args.emb_dim, args.dim, bias=False)
        self.layers = nn.ModuleList([SpectralTransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)
        
        # Init
        nn.init.normal_(self.tok_embeddings.weight, std=0.02)
        nn.init.normal_(self.emb_proj.weight, std=0.02)

    def forward(self, tokens, targets=None):
        bsz, seqlen = tokens.shape
        h = self.emb_proj(self.tok_embeddings(tokens))
        freqs_cis = self.freqs_cis.to(h.device)[:seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).view(1, 1, seqlen, seqlen)

        for layer in self.layers:
            h, _ = layer(h, freqs_cis, mask)

        h_norm = self.norm(h)
        h_small = F.linear(h_norm, self.emb_proj.weight.t()) 
        logits = F.linear(h_small, self.tok_embeddings.weight)
        
        if targets is not None:
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits

# --- Soporte Técnico ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight

def precompute_freqs_cis(dim, end):
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    return torch.outer(torch.arange(end), freqs).float()

def apply_rotary_emb(xq, xk, freqs):
    if freqs is None:
        return xq, xk
    def _reshape(f, x):
        s = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return f.view(*s)
    xq_, xk_ = xq.float().reshape(*xq.shape[:-1], -1, 2), xk.float().reshape(*xk.shape[:-1], -1, 2)
    f = _reshape(freqs, xq_[..., 0])
    cos, sin = torch.cos(f), torch.sin(f)
    xq_out = torch.stack([xq_[...,0]*cos - xq_[...,1]*sin, xq_[...,0]*sin + xq_[...,1]*cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk_[...,0]*cos - xk_[...,1]*sin, xk_[...,0]*sin + xk_[...,1]*cos], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
