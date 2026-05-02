"""
model_auto_analog.py — TinyThinker Auto-Analog-Architect (V175 Era)

Híbrido definitivo:
  - Bancos de Neuronas Analógicas (EXP-9): SUM, PROD, VAR, SIN.
  - Neurogénesis Residual (V170): Crecimiento dinámico de capas al estancarse.
  - Alternancia Lateral (V197): Interacción entre capas analógicas y lógicas.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_lateral_v197 import ResidualLateralBlock

@dataclass
class AutoAnalogArgs:
    dim: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 16384
    multiple_of: int = 256
    ffn_dim_multiplier: float = 2.0
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024

# =============================================================================
# Componentes Base (RMSNorm, RoPE)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == 'privateuseone':
            with torch.autocast('privateuseone', enabled=False):
                x_f = x.float()
                var = (x_f * x_f).mean(-1, keepdim=True)
                return (x_f * torch.rsqrt(var + self.eps)) * self.weight
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    return torch.outer(t, freqs).float()

def _reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs.shape == (x.shape[1], x.shape[-1]), f"{freqs.shape} vs {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    freqs = _reshape_for_broadcast(freqs, xq_[..., 0])
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    xq_0, xq_1 = xq_.unbind(-1)
    xk_0, xk_1 = xk_.unbind(-1)
    xq_out = torch.stack([xq_0*cos - xq_1*sin, xq_0*sin + xq_1*cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk_0*cos - xk_1*sin, xk_0*sin + xk_1*cos], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# =============================================================================
# Bloques Analógicos
# =============================================================================

class AnalogAttention(nn.Module):
    def __init__(self, args: AutoAnalogArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads else args.n_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        nn.init.zeros_(self.wo.weight)

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

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        if past_key_value is not None:
            pk, pv = past_key_value
            xk = torch.cat([pk, xk], dim=2)
            xv = torch.cat([pv, xv], dim=2)

        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask if past_key_value is None and seqlen > 1 else None, 
                                            dropout_p=0.0, is_causal=(past_key_value is None and seqlen > 1))
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out), (xk, xv)

class AnalogFeedForward(nn.Module):
    def __init__(self, args: AutoAnalogArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.bank_dim = hidden_dim // 4
        self.group_size = 2
        
        self.w_linear = nn.Linear(args.dim, self.bank_dim, bias=False)
        self.w_mult   = nn.Linear(args.dim, self.bank_dim * self.group_size, bias=False)
        self.w_var    = nn.Linear(args.dim, self.bank_dim * self.group_size, bias=False)
        self.w_sin    = nn.Linear(args.dim, self.bank_dim, bias=False)
        self.w_out    = nn.Linear(self.bank_dim * 4, args.dim, bias=False)
        
        nn.init.zeros_(self.w_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        out_linear = F.silu(self.w_linear(x))
        
        mult_proj = self.w_mult(x).view(bsz, seq_len, self.bank_dim, self.group_size)
        out_mult = torch.tanh(torch.prod(mult_proj, dim=-1))
        
        var_proj = self.w_var(x).view(bsz, seq_len, self.bank_dim, self.group_size)
        out_var = torch.var(var_proj, dim=-1)
        
        out_sin = torch.sin(self.w_sin(x))
        h_concat = torch.cat([out_linear, out_mult, out_var, out_sin], dim=-1)
        return self.w_out(h_concat)

class ResidualAnalogLayer(nn.Module):
    def __init__(self, args: AutoAnalogArgs):
        super().__init__()
        self.attention = AnalogAttention(args)
        self.feed_forward = AnalogFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_checkpoint = False

    def _fwd(self, x, freqs_cis, mask, past_kv):
        attn_out, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask, past_kv)
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h)), new_kv

    def forward(self, x, freqs_cis, mask, past_key_value=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._fwd, x, freqs_cis, mask, past_key_value, use_reentrant=False)
        return self._fwd(x, freqs_cis, mask, past_key_value)

# =============================================================================
# Auto-Analog Thinker
# =============================================================================

class TinyThinkerAutoAnalog(nn.Module):
    def __init__(self, args: AutoAnalogArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # Comenzamos con 1 capa analógica (Padre)
        self.layers = nn.ModuleList([ResidualAnalogLayer(args)])
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / math.sqrt(args.dim))
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)

    def add_residual_layer(self):
        print(f"🧬 [Auto-Analog] Neurogénesis: Añadiendo Capa {len(self.layers) + 1}...")
        
        # Congelar el pasado (Regla V170)
        for param in self.parameters():
            param.requires_grad = False
            
        # Alternancia Estratégica (Mario V193/V197):
        # Si ya tenemos la capa base (Analógica), la siguiente debe ser Lógica (Lateral).
        if len(self.layers) % 2 == 1:
            print("🧠 [V197] Inyectando CAPA HIJA LATERAL (Razonamiento Simbólico).")
            new_layer = ResidualLateralBlock(self.args.dim)
        else:
            print("📡 [V175] Inyectando CAPA ANALÓGICA (Refinamiento de Rasgos).")
            new_layer = ResidualAnalogLayer(self.args)
            
        # Activar gradientes solo para la nueva capa
        for param in new_layer.parameters():
            param.requires_grad = True
            
        self.layers.append(new_layer)
        print(f"✅ Neurogénesis completada. La red ahora tiene {len(self.layers)} capas.")

    def forward(self, tokens, targets=None, past_key_values=None, use_cache=False):
        _bsz, seqlen = tokens.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None and past_key_values[0] is not None else 0
        
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(h.device)[past_len:past_len + seqlen]
        mask = _causal_mask(seqlen, tokens.device, h.dtype) if past_key_values is None else None

        new_kvs = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ResidualAnalogLayer):
                pkv = past_key_values[i] if past_key_values is not None and i < len(past_key_values) else None
                h, kv = layer(h, freqs_cis, mask, pkv)
                new_kvs.append(kv)
            else:
                # Capa Lateral (V197)
                h = layer(h)
                new_kvs.append(None) # Sin cache KV para capas lógicas

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
