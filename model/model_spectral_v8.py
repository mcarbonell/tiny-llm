"""
model_spectral_v8.py — The Holographic MoE Thinker
Fusión total de Attention-Neuron y TinyThinker.

Innovaciones:
1. Holographic Attention: Memoria O(1) mediante acumuladores espaciotemporales (Roll).
2. Extreme MoE: 131k expertos activados por resonancia espectral.
3. Zero-KV-Cache: No más descompresión JPEG, solo resonancia pura.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SpectralArgs:
    dim: int = 1024
    emb_dim: int = 128
    n_layers: int = 6
    n_heads: int = 16       # Añadido para compatibilidad
    n_kv_heads: int = 4     # Añadido para compatibilidad
    vocab_size: int = 32768
    # MoE Config
    num_experts: int = 131072
    top_k: int = 16
    # Spectral Projections
    k_dim: int = 128
    max_batch_size: int = 32
    max_seq_len: int = 1024

# --- TRANSFORMADA DE WALSH-HADAMARD RÁPIDA (FWHT) ---
def fwht(x):
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- CAPA MOE EXTREMA (Basada en V163d) ---
class ExtremeSpectralMoE(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.dim = args.dim
        
        # Firmas de expertos (Knowledge Bank)
        self.signatures = nn.Parameter(torch.randn(args.num_experts, args.dim) * 0.02)
        # Votos de los expertos
        self.expert_weights = nn.Parameter(torch.randn(args.num_experts, args.dim) * 0.01)

    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        x_spec = fwht(x_flat)
        
        # Gating por resonancia
        scores = torch.mm(x_spec, F.normalize(self.signatures, p=2, dim=1).t())
        
        # Selección Top-K
        top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=1)
        weights = F.softmax(top_scores * 5.0, dim=1)
        
        # Mezcla de expertos
        out_experts = self.expert_weights[top_indices]
        out = (out_experts * weights.unsqueeze(-1)).sum(dim=1)
        
        return out.view(b, t, d)

# --- ATENCIÓN HOLOGRÁFICA (Basada en V163e) ---
class HolographicAttention(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.dim = args.dim
        self.q_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.k_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.v_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.o_proj = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x, hologram=None, pos=0):
        # Soporte para secuencias (T > 1) para entrenamiento
        b, t, d = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        outputs = []
        curr_hologram = hologram if hologram is not None else torch.zeros(b, d, device=x.device)
        
        for i in range(t):
            qi = fwht(q[:, i, :])
            ki = fwht(k[:, i, :])
            vi = fwht(v[:, i, :])
            
            # Multiplexado temporal (Roll dependiente de la posición absoluta)
            abs_pos = pos + i
            shifted_k = torch.roll(ki, shifts=abs_pos % self.dim, dims=1)
            
            # Actualización recursiva
            curr_hologram = F.normalize(curr_hologram + (shifted_k * vi), p=2, dim=1)
            
            # Recall por resonancia
            recall = qi * curr_hologram
            outputs.append(fwht(recall))
            
        out = torch.stack(outputs, dim=1)
        return self.o_proj(out), curr_hologram

# --- BLOQUE TRANSFORMER V8 ---
class SpectralV8Block(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.hra = HolographicAttention(args)
        self.moe = ExtremeSpectralMoE(args)
        self.norm1 = nn.RMSNorm(args.dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(args.dim)
        self.norm2 = nn.RMSNorm(args.dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(args.dim)

    def forward(self, x, hologram=None, pos=0):
        h_attn, new_hologram = self.hra(self.norm1(x), hologram, pos)
        x = x + h_attn
        x = x + self.moe(self.norm2(x))
        return x, new_hologram

# --- EL PENSADOR V8 ---
class SpectralThinkerV8(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.emb_dim)
        self.emb_proj = nn.Linear(args.emb_dim, args.dim, bias=False)
        
        self.layers = nn.ModuleList([SpectralV8Block(args) for _ in range(args.n_layers)])
        self.norm_final = nn.RMSNorm(args.dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(args.dim)

        # --- INICIALIZACIÓN DE ESTABILIDAD ---
        nn.init.normal_(self.tok_embeddings.weight, std=0.02)
        nn.init.normal_(self.emb_proj.weight, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens, targets=None, holograms=None, pos=0, use_cache=False):
        b, t = tokens.shape
        h = self.emb_proj(self.tok_embeddings(tokens))
        
        new_holograms = []
        for i, layer in enumerate(self.layers):
            prev_h = holograms[i] if holograms is not None else None
            h, new_h = layer(h, prev_h, pos)
            new_holograms.append(new_h)
            
        h_final = self.norm_final(h)
        h_small = F.linear(h_final, self.emb_proj.weight.t())
        logits = F.linear(h_small, self.tok_embeddings.weight)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
            
        return (logits, new_holograms) if use_cache else logits

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Sincronizamos el peso con el dtype de la entrada (evita el mismatch bfloat16/float32)
        w = self.weight.to(x.dtype)
        # Intentamos usar la función optimizada de PyTorch si existe (disponible en versiones recientes)
        if hasattr(F, 'rms_norm'):
            return F.rms_norm(x, (x.shape[-1],), w, self.eps)
        # Fallback manual optimizado
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * w

# Inyectamos nuestra versión mejorada
nn.RMSNorm = RMSNorm
