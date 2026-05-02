"""
model_auto_architect.py — TinyThinker Auto-Architect (V170 Era)

Implementación del concepto "Neurogénesis Residual" (V167-V170).
La red comienza con una sola capa (Pensamiento Rápido). Si el Loss se estanca,
un orquestador externo puede llamar al método `add_residual_layer()`.
Esto congela las capas existentes y añade una nueva capa que aprende
exclusivamente a corregir el residuo (el error) de las anteriores, sin olvido catastrófico.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_spectral_v4 import _get_dct_rows, _get_walsh_rows, _next_pow2, DCTLinear, WalshLinear, RMSNorm, apply_rotary_emb, precompute_freqs_cis

@dataclass
class AutoArchitectArgs:
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 16384
    multiple_of: int = 256
    ffn_dim_multiplier: float = 2.0
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024
    
    # Spectral Params
    k_dim_attn: int = 64
    k_dim_ffn: int = 64
    k_hidden_ffn: int = 128

# =============================================================================
# Componentes Espectrales (Atención y FFN)
# =============================================================================

class SpectralAttention(nn.Module):
    def __init__(self, args: AutoArchitectArgs):
        super().__init__()
        self.n_heads    = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads else args.n_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_rep    = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        k = args.k_dim_attn

        self.wq = DCTLinear(args.dim, args.n_heads    * self.head_dim, k_in=k, k_out=k)
        self.wk = DCTLinear(args.dim, self.n_kv_heads * self.head_dim, k_in=k, k_out=k)
        self.wv = DCTLinear(args.dim, self.n_kv_heads * self.head_dim, k_in=k, k_out=k)
        self.wo = DCTLinear(args.n_heads * self.head_dim, args.dim,    k_in=k, k_out=k)
        
        # Inicialización a 0 en la proyección de salida para Neurogénesis Residual perfecta
        nn.init.zeros_(self.wo.core)

    def forward(self, x, freqs_cis, mask, past_key_value=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads,    self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs=freqs_cis)

        if self.n_rep > 1:
            xk = xk[:,:,:,None,:].expand(bsz,seqlen,self.n_kv_heads,self.n_rep,self.head_dim).flatten(2,3)
            xv = xv[:,:,:,None,:].expand(bsz,seqlen,self.n_kv_heads,self.n_rep,self.head_dim).flatten(2,3)

        xq, xk, xv = xq.transpose(1,2), xk.transpose(1,2), xv.transpose(1,2)

        if past_key_value is not None:
            pk, pv = past_key_value
            xk = torch.cat([pk, xk], dim=2)
            xv = torch.cat([pv, xv], dim=2)

        if past_key_value is None and seqlen > 1:
            out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0, is_causal=False)

        out = out.transpose(1,2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out), (xk, xv)


class SpectralFeedForward(nn.Module):
    def __init__(self, args: AutoArchitectArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        dim_w    = _next_pow2(args.dim)
        hidden_w = _next_pow2(hidden_dim)
        k_d, k_h = args.k_dim_ffn, args.k_hidden_ffn

        self.w1 = WalshLinear(dim_w, hidden_w, k_in=k_d, k_out=k_h)
        self.w2 = WalshLinear(hidden_w, dim_w,  k_in=k_h, k_out=k_d)
        self.w3 = WalshLinear(dim_w, hidden_w, k_in=k_d, k_out=k_h)

        self.in_dim, self.out_dim = args.dim, args.dim
        self.dim_w = dim_w
        
        # Inicialización a 0 en la proyección de salida para Neurogénesis Residual perfecta
        nn.init.zeros_(self.w2.core)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_dim != self.dim_w:
            x = F.pad(x, (0, self.dim_w - self.in_dim))
        h = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.out_dim != self.dim_w:
            h = h[..., :self.out_dim]
        return h


class ResidualSpecialistLayer(nn.Module):
    """Una capa que nace para corregir el error (residuo) de las anteriores."""
    def __init__(self, args: AutoArchitectArgs):
        super().__init__()
        self.attention      = SpectralAttention(args)
        self.feed_forward   = SpectralFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm       = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_checkpoint = False
        
        # En V170, la capa comienza devolviendo EXACTAMENTE 0.
        # h_out = h_in + 0. Así el paso inicial de esta capa nueva no rompe el loss
        # que las capas anteriores ya habían optimizado.
        
    def _fwd(self, x, freqs_cis, mask, past_kv):
        attn_out, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask, past_kv)
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h)), new_kv

    def forward(self, x, freqs_cis, mask, past_key_value=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._fwd, x, freqs_cis, mask, past_key_value, use_reentrant=False)
        return self._fwd(x, freqs_cis, mask, past_key_value)

# =============================================================================
# Auto-Architect Thinker
# =============================================================================

class TinyThinkerAutoArchitect(nn.Module):
    def __init__(self, args: AutoArchitectArgs):
        super().__init__()
        self.args       = args
        self.vocab_size = args.vocab_size
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # Comenzamos con UNA SOLA CAPA (Pensamiento Rápido)
        self.layers = nn.ModuleList([ResidualSpecialistLayer(args)])
        
        self.norm   = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / math.sqrt(args.dim))
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)

    def add_residual_layer(self):
        """
        V167-V170: Neurogénesis Residual.
        Congela todas las capas existentes (y el embedding/output) para que
        no sufran olvido catastrófico. Instancia una nueva capa especializada
        que aprenderá a corregir los errores de las anteriores.
        """
        print(f"🌱 [Auto-Architect] Cultivando Capa {len(self.layers) + 1}...")
        
        # 1. Congelar el pasado
        for param in self.parameters():
            param.requires_grad = False
            
        # 2. Hacer nacer la nueva capa
        new_layer = ResidualSpecialistLayer(self.args)
        
        # Activar gradientes solo para la nueva capa
        for param in new_layer.parameters():
            param.requires_grad = True
            
        self.layers.append(new_layer)
        print(f"✅ Capa {len(self.layers)} activa. Parámetros entrenables ahora limitados a esta capa.")

    def forward(self, tokens, targets=None, past_key_values=None, use_cache=False):
        _bsz, seqlen = tokens.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if tokens.device.type == 'privateuseone':
            with torch.autocast('privateuseone', enabled=False):
                h = self.tok_embeddings(tokens)
                freqs_cis = self.freqs_cis.to(h.device)[past_len:past_len + seqlen]
                mask = _causal_mask(seqlen, tokens.device, h.dtype) if past_key_values is None else None
        else:
            h = self.tok_embeddings(tokens)
            freqs_cis = self.freqs_cis.to(h.device)[past_len:past_len + seqlen]
            mask = _causal_mask(seqlen, tokens.device, h.dtype) if past_key_values is None else None

        new_kvs: List = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None and i < len(past_key_values) else None
            h, kv = layer(h, freqs_cis, mask, pkv)
            new_kvs.append(kv)

        logits = self.output(self.norm(h))

        if targets is not None:
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if use_cache:
            return logits, new_kvs
        return logits

def _causal_mask(seqlen, device, dtype):
    if seqlen <= 1: return None
    mask = torch.zeros(seqlen, seqlen, device=device, dtype=dtype)
    mask.masked_fill_(torch.ones(seqlen, seqlen, device=device, dtype=torch.bool).tril().logical_not(), float('-inf'))
    return mask.view(1, 1, seqlen, seqlen)

def count_params(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'trainable': trainable, 'trainable_M': round(trainable / 1e6, 2)}
