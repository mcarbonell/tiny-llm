"""
model_coga_spectral.py — COGA + Spectral Architecture

Fusión del "Cognitive Operating System Architecture" (COGA) con 
la arquitectura Matrix-Free Espectral (DCT Attention + Walsh FFN) v4.

Componentes clave:
  - N capas 'pre' (SSM/Parsing - implementado como Spectral Dense ligero)
  - N capas 'core' (Razonamiento profundo con recurrencia)
  - N capas 'post' (Refinamiento y Output)
  - Scratchpad Mutable (Cross-Attention)
  - Compresión Espectral DCT en Atención
  - Compresión Espectral Walsh en Feed-Forward (MoE)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_spectral_v4 import _get_dct_rows, _get_walsh_rows, _next_pow2

# =============================================================================
# 1. CogaSpectralArgs
# =============================================================================

@dataclass
class CogaSpectralArgs:
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
    
    # MoE Args
    n_experts: int = 8
    top_k: int = 2
    n_reserved: int = 4
    
    # COGA Scratchpad Args
    n_scratch_slots: int = 32
    
    # COGA Recurrence Args
    n_pre_layers: int = 2
    n_core_layers: int = 4
    n_post_layers: int = 2
    max_recurrence_steps: int = 4


# =============================================================================
# 2. Capas Espectrales Base (Matrix-Free)
# =============================================================================

class DCTLinear(nn.Module):
    """Proyección espectral DCT. Forward matrix-free."""
    def __init__(self, in_features: int, out_features: int,
                 k_in: int, k_out: int, bias: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.k_in  = min(k_in,  in_features)
        self.k_out = min(k_out, out_features)

        self._D_in_k_cpu  = _get_dct_rows(in_features,  self.k_in)
        self._D_out_k_cpu = _get_dct_rows(out_features, self.k_out)
        self._D_in_k_dev  = None
        self._D_out_k_dev = None
        self._cached_dev  = None

        self.core = nn.Parameter(
            torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in))
        )
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

    def _get_bases(self, device):
        if self._cached_dev != device:
            self._D_in_k_dev  = self._D_in_k_cpu.to(device)
            self._D_out_k_dev = self._D_out_k_cpu.to(device)
            self._cached_dev  = device
        return self._D_in_k_dev, self._D_out_k_dev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D_in_k, D_out_k = self._get_bases(x.device)
        h = x      @ D_in_k.t()
        h = h      @ self.core.t()
        h = h      @ D_out_k
        if self.bias_param is not None:
            h = h + self.bias_param
        return h

class WalshLinear(nn.Module):
    """Proyección espectral Walsh-Hadamard. Forward matrix-free."""
    def __init__(self, in_features: int, out_features: int,
                 k_in: int, k_out: int, bias: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.k_in  = min(k_in,  in_features)
        self.k_out = min(k_out, out_features)

        self._H_in_k_cpu  = _get_walsh_rows(in_features,  self.k_in)
        self._H_out_k_cpu = _get_walsh_rows(out_features, self.k_out)
        self._H_in_k_dev  = None
        self._H_out_k_dev = None
        self._cached_dev  = None

        self.core = nn.Parameter(
            torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in))
        )
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

    def _get_bases(self, device):
        if self._cached_dev != device:
            self._H_in_k_dev  = self._H_in_k_cpu.to(device)
            self._H_out_k_dev = self._H_out_k_cpu.to(device)
            self._cached_dev  = device
        return self._H_in_k_dev, self._H_out_k_dev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in_k, H_out_k = self._get_bases(x.device)
        h = x @ H_in_k.t()
        h = h @ self.core.t()
        h = h @ H_out_k
        if self.bias_param is not None:
            h = h + self.bias_param
        return h


# =============================================================================
# 3. Componentes Estándar (Norm, RoPE)
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

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
# 4. Atención Espectral y Cross-Attention COGA
# =============================================================================

class SpectralAttention(nn.Module):
    """Self-Attention usando DCTLinear (Compresión semántica suave)."""
    def __init__(self, args: CogaSpectralArgs):
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

class SpectralCrossAttention(nn.Module):
    """Cross-Attention con el Scratchpad Mutable usando DCTLinear."""
    def __init__(self, args: CogaSpectralArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        k = args.k_dim_attn
        
        self.wq = DCTLinear(args.dim, args.n_heads * self.head_dim, k_in=k, k_out=k)
        self.wk = DCTLinear(args.dim, args.n_heads * self.head_dim, k_in=k, k_out=k)
        self.wv = DCTLinear(args.dim, args.n_heads * self.head_dim, k_in=k, k_out=k)
        self.wo = DCTLinear(args.n_heads * self.head_dim, args.dim, k_in=k, k_out=k)

    def forward(self, query: torch.Tensor, scratchpad: torch.Tensor):
        bsz, seqlen, _ = query.shape
        _, slots, _ = scratchpad.shape
        
        xq = self.wq(query).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(scratchpad).view(bsz, slots, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(scratchpad).view(bsz, slots, self.n_heads, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


# =============================================================================
# 5. Spectral MoE Feed-Forward (Walsh)
# =============================================================================

class SpectralExpert(nn.Module):
    """Un experto MoE basado en Transformada de Walsh (lógica binaria afilada)."""
    def __init__(self, dim: int, hidden_dim: int, args: CogaSpectralArgs):
        super().__init__()
        dim_w    = _next_pow2(dim)
        hidden_w = _next_pow2(hidden_dim)
        k_d, k_h = args.k_dim_ffn, args.k_hidden_ffn

        self.w1 = WalshLinear(dim_w, hidden_w, k_in=k_d, k_out=k_h)
        self.w2 = WalshLinear(hidden_w, dim_w,  k_in=k_h, k_out=k_d)
        self.w3 = WalshLinear(dim_w, hidden_w, k_in=k_d, k_out=k_h)

        self.in_dim, self.out_dim = dim, dim
        self.dim_w = dim_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_dim != self.dim_w:
            x = F.pad(x, (0, self.dim_w - self.in_dim))
        h = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.out_dim != self.dim_w:
            h = h[..., :self.out_dim]
        return h

class SpectralMoEFeedForward(nn.Module):
    def __init__(self, args: CogaSpectralArgs):
        super().__init__()
        self.dim = args.dim
        self.n_experts = args.n_experts
        self.top_k = args.top_k
        self.n_reserved = args.n_reserved
        
        # El enrutador se mantiene denso porque es muy pequeño (dim -> n_experts)
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        
        hidden_dim = int(2 * (4 * args.dim) / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.experts = nn.ModuleList([
            SpectralExpert(dim=args.dim, hidden_dim=hidden_dim, args=args)
            for _ in range(args.n_experts)
        ])

    def forward(self, x: torch.Tensor, train_reserved: bool = False):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        gate_logits = self.gate(x_flat)
        if not train_reserved and self.n_reserved > 0:
            mask = torch.zeros_like(gate_logits)
            mask[:, -self.n_reserved:] = float('-inf')
            gate_logits = gate_logits + mask
            
        weights = F.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            token_indices, k_indices = (top_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                expert_out = self.experts[i](x_flat[token_indices])
                out[token_indices] += top_weights[token_indices, k_indices].unsqueeze(-1) * expert_out
                
        return out.view(batch_size, seq_len, dim)


# =============================================================================
# 6. Spectral Transformer Block
# =============================================================================

class SpectralTransformerBlock(nn.Module):
    def __init__(self, args: CogaSpectralArgs, use_cross_attention: bool = True):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        
        self.attention = SpectralAttention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        if self.use_cross_attention:
            self.cross_attention = SpectralCrossAttention(args)
            self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            
        self.feed_forward = SpectralMoEFeedForward(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_checkpoint = False

    def _fwd(self, x, freqs_cis, mask, scratchpad, past_kv, train_reserved):
        attn_out, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask, past_kv)
        h = x + attn_out
        
        if self.use_cross_attention and scratchpad is not None:
            cross_out = self.cross_attention(self.cross_attention_norm(h), scratchpad)
            h = h + cross_out
            
        ffn_out = self.feed_forward(self.ffn_norm(h), train_reserved=train_reserved)
        return h + ffn_out, new_kv

    def forward(self, x, freqs_cis, mask, scratchpad=None, past_key_value=None, use_cache=False, train_reserved=False):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._fwd, x, freqs_cis, mask, scratchpad, past_key_value, train_reserved, use_reentrant=False)
        
        out, new_kv = self._fwd(x, freqs_cis, mask, scratchpad, past_key_value, train_reserved)
        if use_cache or past_key_value is not None:
            return out, new_kv
        return out


# =============================================================================
# 7. TinyThinker COGA Espectral — El Sistema Cognitivo Completo
# =============================================================================

class TinyThinkerCogaSpectral(nn.Module):
    def __init__(self, args: CogaSpectralArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.dim = args.dim
        self.n_scratch_slots = args.n_scratch_slots
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # Bloque A: Parsing (SSM-like, pero implementado como Transformer ligero sin Cross-Attention)
        self.pre_layers = nn.ModuleList([
            SpectralTransformerBlock(args, use_cross_attention=False) 
            for _ in range(args.n_pre_layers)
        ])
        
        # Bloque B: Razonamiento Profundo (Con Scratchpad Cross-Attention y Recurrencia)
        self.core_layers = nn.ModuleList([
            SpectralTransformerBlock(args, use_cross_attention=True) 
            for _ in range(args.n_core_layers)
        ])
        
        # Bloque C: Refinamiento Output
        self.post_layers = nn.ModuleList([
            SpectralTransformerBlock(args, use_cross_attention=False) 
            for _ in range(args.n_post_layers)
        ])
        
        # Halt Head: Determina dinámicamente si seguir iterando en el Bloque B
        self.halt_head = nn.Linear(args.dim, 1)
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Como aprendimos en v3, NO forzamos weight tying por defecto en modelos pequeños
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / math.sqrt(args.dim))

        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len * 2
        )

    def forward(self, tokens, targets=None, scratchpad=None, past_key_values=None, use_cache=False, train_reserved=False):
        _bsz, seqlen = tokens.shape
        device = tokens.device

        # Autocast workaround
        if device.type == 'privateuseone':
            with torch.autocast('privateuseone', enabled=False):
                h = self.tok_embeddings(tokens)
                freqs_cis = self.freqs_cis.to(h.device)[:seqlen]
                mask = _causal_mask(seqlen, device, h.dtype)
        else:
            h = self.tok_embeddings(tokens)
            freqs_cis = self.freqs_cis.to(h.device)[:seqlen]
            mask = _causal_mask(seqlen, device, h.dtype)

        # Inicilizar scratchpad dinámico
        if scratchpad is None:
            scratchpad = torch.zeros(_bsz, self.n_scratch_slots, self.dim, device=device, dtype=h.dtype)

        past_key_values_out = []
        layer_idx = 0
        
        # --- BLOQUE A: Pre-Layers ---
        for layer in self.pre_layers:
            pkv = past_key_values[layer_idx] if past_key_values else None
            h, kv = layer(h, freqs_cis, mask, None, past_key_value=pkv, use_cache=True, train_reserved=train_reserved)
            past_key_values_out.append(kv)
            layer_idx += 1
            
        # --- CEREBELO ESPECTRAL (Early Exit Logic) ---
        # Calculamos la probabilidad de "Halt" antes del pesado Bloque B
        halt_logits = self.halt_head(h[:, -1:, :])
        halt_prob = torch.sigmoid(halt_logits).squeeze(-1)
        
        steps_to_run = self.args.max_recurrence_steps
        if not self.training and _bsz == 1:
            # Si el modelo está "seguro" (halt_prob ~ 1.0), steps_to_run se acerca a 0/1.
            # Esto es inferencia dinámica guiada por entropía (V89)
            estimated_steps = max(1, round((1.0 - halt_prob.item()) * self.args.max_recurrence_steps))
            steps_to_run = estimated_steps
            
        # --- BLOQUE B: Core-Layers Recurrentes ---
        for step in range(steps_to_run):
            core_idx_start = layer_idx
            for layer in self.core_layers:
                is_last_step = (step == steps_to_run - 1)
                pkv = past_key_values[core_idx_start] if past_key_values else None
                
                h, kv = layer(h, freqs_cis, mask, scratchpad, past_key_value=pkv, 
                              use_cache=True, train_reserved=train_reserved)
                
                if is_last_step:
                    past_key_values_out.append(kv)
                core_idx_start += 1
                
        layer_idx += self.args.n_core_layers
        
        # --- BLOQUE C: Post-Layers ---
        for layer in self.post_layers:
            pkv = past_key_values[layer_idx] if past_key_values else None
            h, kv = layer(h, freqs_cis, mask, None, past_key_value=pkv, use_cache=True, train_reserved=train_reserved)
            past_key_values_out.append(kv)
            layer_idx += 1

        # Output
        logits = self.output(self.norm(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
            
        if use_cache:
            return logits, past_key_values_out
        return logits


def _causal_mask(seqlen, device, dtype):
    if seqlen <= 1:
        return None
    mask = torch.zeros(seqlen, seqlen, device=device, dtype=dtype)
    mask.masked_fill_(
        torch.ones(seqlen, seqlen, device=device, dtype=torch.bool).tril().logical_not(),
        float('-inf')
    )
    return mask.view(1, 1, seqlen, seqlen)

def count_params(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'trainable': trainable, 'trainable_M': round(trainable / 1e6, 2)}
