"""
model_spectral_v6.py — SpectralThinker v6 (Causal-JPEG / Strict Causal Memory)

Cambios vs v5:
  - ELIMINADO EL CAUSAL LEAKAGE: El entrenamiento es ahora 100% causal y nítido.
  - El modelo entrena con atención estándar (sharp) para aprender rasgos reales.
  - La compresión temporal se activa solo en Inferencia (generación incremental),
    asegurando que solo se comprima el pasado real generado.
  - Mantiene pesos Matrix-Free (DCT/Walsh) para eficiencia en RAM.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SpectralArgs:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 16384
    multiple_of: int = 256
    ffn_dim_multiplier: float = 2.0
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024
    
    # Compresión de Pesos
    k_dim_attn: int = 64
    k_dim_ffn: int = 64
    k_hidden_ffn: int = 128
    
    # Compresión de Cache (Solo en inferencia)
    k_seq_len: int = 64 

# =============================================================================
# 1. Utilidades Espectrales
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
    assert N > 0 and (N & (N - 1)) == 0
    k = min(k, N)
    key = (N, k)
    if key not in _WALSH_ROW_CACHE:
        _WALSH_ROW_CACHE[key] = _walsh_rows_recursive(N, k)
    return _WALSH_ROW_CACHE[key]

def _walsh_rows_recursive(N: int, k: int) -> torch.Tensor:
    if N == k:
        H = torch.tensor([[1.0]])
        while H.shape[0] < N:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        return H / math.sqrt(N)
    if N == 1: return torch.tensor([[1.0]])
    half = N // 2
    k_half = min(k, half)
    small = _walsh_rows_recursive(half, k_half)
    top = torch.cat([small, small], dim=1) / math.sqrt(2)
    if k <= half: return top
    extra_k = k - half
    extra = _walsh_rows_recursive(half, extra_k)
    bottom = torch.cat([extra, -extra], dim=1) / math.sqrt(2)
    return torch.cat([top, bottom], dim=0)

# =============================================================================
# 2. Capas Espectrales Matrix-Free
# =============================================================================

class DCTLinear(nn.Module):
    def __init__(self, in_features, out_features, k_in, k_out, bias=False):
        super().__init__()
        self.k_in, self.k_out = min(k_in, in_features), min(k_out, out_features)
        self.in_features, self.out_features = in_features, out_features
        self.core = nn.Parameter(torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        d_in = _get_dct_rows(self.in_features, self.k_in).to(x.device).to(x.dtype)
        d_out = _get_dct_rows(self.out_features, self.k_out).to(x.device).to(x.dtype)
        h = x @ d_in.t()
        h = h @ self.core.t()
        h = h @ d_out
        return h + self.bias if self.bias is not None else h

class WalshLinear(nn.Module):
    def __init__(self, in_features, out_features, k_in, k_out, bias=False):
        super().__init__()
        self.k_in, self.k_out = min(k_in, in_features), min(k_out, out_features)
        self.in_features, self.out_features = in_features, out_features
        self.core = nn.Parameter(torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        h_in = _get_walsh_rows(self.in_features, self.k_in).to(x.device).to(x.dtype)
        h_out = _get_walsh_rows(self.out_features, self.k_out).to(x.device).to(x.dtype)
        h = x @ h_in.t()
        h = h @ self.core.t()
        h = h @ h_out
        return h + self.bias if self.bias is not None else h

# =============================================================================
# 3. Causal Spectral Attention (V6)
# =============================================================================

class SpectralAttentionV6(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads or args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.k_seq_len = args.k_seq_len

        self.wq = DCTLinear(args.dim, args.n_heads * self.head_dim, args.k_dim_attn, args.k_dim_attn)
        self.wk = DCTLinear(args.dim, self.n_kv_heads * self.head_dim, args.k_dim_attn, args.k_dim_attn)
        self.wv = DCTLinear(args.dim, self.n_kv_heads * self.head_dim, args.k_dim_attn, args.k_dim_attn)
        self.wo = DCTLinear(args.n_heads * self.head_dim, args.dim, args.k_dim_attn, args.k_dim_attn)

    def _compress_past(self, tensor, device):
        """Comprime el pasado (temporal) usando DCT."""
        bsz, seq_len, heads, h_dim = tensor.shape
        if seq_len <= self.k_seq_len: return tensor
        d_seq = _get_dct_rows(seq_len, self.k_seq_len).to(device).to(tensor.dtype)
        # (bsz, heads, h_dim, seq_len) @ (seq_len, k_seq)
        return (tensor.permute(0, 2, 3, 1) @ d_seq.t()).permute(0, 3, 1, 2)

    def _decompress_past(self, compressed, target_len, device):
        """Reconstruye el pasado borroso."""
        if compressed.shape[1] == target_len: return compressed
        d_seq = _get_dct_rows(target_len, self.k_seq_len).to(device).to(compressed.dtype)
        # (bsz, heads, h_dim, k_seq) @ (k_seq, target_len)
        return (compressed.permute(0, 2, 3, 1) @ d_seq).permute(0, 3, 1, 2)

    def forward(self, x, freqs_cis, mask, past_key_value=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs=freqs_cis)

        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim).flatten(2, 3)
            xv = xv[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim).flatten(2, 3)

        # MODO INFERENCIA (GENERACIÓN TOKEN A TOKEN)
        if past_key_value is not None:
            pk_comp, pv_comp, past_len = past_key_value
            # Descomprimir solo lo que es estrictamente pasado
            pk = self._decompress_past(pk_comp, past_len, x.device)
            pv = self._decompress_past(pv_comp, past_len, x.device)
            
            # Concatenar presente nítido con pasado borroso
            xk_full = torch.cat([pk, xk], dim=1)
            xv_full = torch.cat([pv, xv], dim=1)
            
            # Guardar nueva versión comprimida
            new_past_len = past_len + seqlen
            new_kv = (self._compress_past(xk_full, x.device), self._compress_past(xv_full, x.device), new_past_len)
            
            out = F.scaled_dot_product_attention(xq.transpose(1, 2), xk_full.transpose(1, 2), xv_full.transpose(1, 2), is_causal=False)
        
        # MODO ENTRENAMIENTO (BLOQUE COMPLETO)
        else:
            # NO APLICAMOS COMPRESIÓN AL BLOQUE DE ENTRENAMIENTO PARA EVITAR CAUSAL LEAKAGE
            # El modelo aprende con KV nítidos. 
            # (Opcional: inyectar ruido aquí para robustez, pero sin leak).
            out = F.scaled_dot_product_attention(xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2), attn_mask=mask, is_causal=False)
            new_kv = (self._compress_past(xk, x.device), self._compress_past(xv, x.device), seqlen)

        return self.wo(out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)), new_kv

# =============================================================================
# 4. Arquitectura de Soporte (RMSNorm, RoPE, FFN, Block)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight

def apply_rotary_emb(xq, xk, freqs):
    def _reshape(f, x):
        s = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return f.view(*s)
    xq_, xk_ = xq.float().reshape(*xq.shape[:-1], -1, 2), xk.float().reshape(*xk.shape[:-1], -1, 2)
    f = _reshape(freqs, xq_[..., 0])
    cos, sin = torch.cos(f), torch.sin(f)
    xq_out = torch.stack([xq_[...,0]*cos - xq_[...,1]*sin, xq_[...,0]*sin + xq_[...,1]*cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk_[...,0]*cos - xk_[...,1]*sin, xk_[...,0]*sin + xk_[...,1]*cos], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SpectralFeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = int(2 * (4 * args.dim) / 3)
        hidden = args.multiple_of * ((hidden + args.multiple_of - 1) // args.multiple_of)
        dim_w, hidden_w = 1 << (args.dim-1).bit_length(), 1 << (hidden-1).bit_length()
        self.w1 = WalshLinear(dim_w, hidden_w, args.k_dim_ffn, args.k_hidden_ffn)
        self.w2 = WalshLinear(hidden_w, dim_w, args.k_hidden_ffn, args.k_dim_ffn)
        self.w3 = WalshLinear(dim_w, hidden_w, args.k_dim_ffn, args.k_hidden_ffn)
        self.dim, self.dim_w = args.dim, dim_w
    def forward(self, x):
        if self.dim != self.dim_w: x = F.pad(x, (0, self.dim_w - self.dim))
        h = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return h[..., :self.dim] if self.dim != self.dim_w else h

class SpectralTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = SpectralAttentionV6(args)
        self.feed_forward = SpectralFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    def forward(self, x, freqs_cis, mask, past_kv=None):
        h_attn, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask, past_kv)
        h = x + h_attn
        return h + self.feed_forward(self.ffn_norm(h)), new_kv

# =============================================================================
# 5. Modelo Final (TinyThinker Spectral V6)
# =============================================================================

class SpectralThinker(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([SpectralTransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / math.sqrt(args.dim))
        
        # Precompute RoPE freqs
        dim_rope = args.dim // args.n_heads
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, dim_rope, 2)[:(dim_rope // 2)].float() / dim_rope))
        t = torch.arange(args.max_seq_len * 2)
        self.freqs_cis = torch.outer(t, freqs).float()

    def forward(self, tokens, targets=None, past_key_values=None, use_cache=False):
        _bsz, seqlen = tokens.shape
        past_len = past_key_values[0][2] if past_key_values is not None else 0
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(h.device)[past_len:past_len + seqlen]
        mask = None
        if past_key_values is None and seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).view(1, 1, seqlen, seqlen)

        new_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            h, kv = layer(h, freqs_cis, mask, pkv)
            new_kvs.append(kv)

        logits = self.output(self.norm(h))
        if targets is not None:
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return (logits, new_kvs) if use_cache else logits

def _causal_mask(seqlen, device, dtype):
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1).view(1, 1, seqlen, seqlen)
