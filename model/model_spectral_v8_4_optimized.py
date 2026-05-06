"""
model_spectral_v8_4_optimized.py — The "Ghost in the Machine" Architecture
Optimized for Speed: Minimal Domain Switching & Compiled FWHT.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SpectralArgs:
    dim: int = 32768
    emb_dim: int = 128
    n_layers: int = 16
    vocab_size: int = 32768
    num_experts: int = 128
    top_k: int = 8
    max_batch_size: int = 32
    max_seq_len: int = 4096

# --- TRANSFORMADA RÁPIDA OPTIMIZADA ---
def fwht_iterative(x):
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.stack([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# Intentamos compilar pero con fallback robusto para Windows sin Visual Studio
try:
    import sys
    # En Windows, torch.compile requiere cl.exe (Visual Studio) que a menudo no está en el PATH
    if hasattr(torch, "compile") and sys.platform != "win32":
        fwht = torch.compile(fwht_iterative, dynamic=True)
    else:
        fwht = fwht_iterative
except Exception:
    fwht = fwht_iterative

class SpectralRMSNorm(nn.Module):
    """
    RMSNorm en el dominio Espectral.
    Gracias al Teorema de Parseval, la energía media es la misma en ambos dominios.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x_spec):
        # x_spec: (B, T, D)
        # La norma RMS es invariante ante FWHT (ortogonal)
        rms = torch.rsqrt(x_spec.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_spec * rms) * self.weight

class SpectralLinear(nn.Module):
    """Filtro Diagonal que asume entrada y salida ESPECTRAL."""
    def __init__(self, dim):
        super().__init__()
        self.diag = nn.Parameter(torch.randn(dim) * 0.02)
    def forward(self, x_spec):
        return x_spec * self.diag

class ResonantSpectralMoE_v84(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.dim = args.dim
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.expert_signatures = nn.Parameter(torch.randn(args.num_experts, args.dim) * 0.02)
        self.expert_filters = nn.Parameter(torch.randn(args.num_experts, args.dim) * 0.02)

    def forward(self, x_spec):
        b, t, d = x_spec.shape
        x_flat = x_spec.view(-1, d)
        
        # GATING (Ya estamos en spectral, ahorramos 1 FWHT)
        logits = torch.mm(x_flat, F.normalize(self.expert_signatures, p=2, dim=1).t())
        scores, indices = torch.topk(logits, self.top_k, dim=1)
        weights = F.softmax(scores * 5.0, dim=1)
        
        out_spec = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            out_spec += w * (x_flat * self.expert_filters[idx])
            
        return out_spec.view(b, t, d)

class SpectralHolographicAttention_v84(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.dim = args.dim
        # Filtros Diagonales (Operan en el espectro)
        self.q_filter = SpectralLinear(args.dim)
        self.k_filter = SpectralLinear(args.dim)
        self.v_filter = SpectralLinear(args.dim)
        self.o_filter = SpectralLinear(args.dim)
        self.saliency_vec = nn.Parameter(torch.ones(args.dim))

    def forward(self, x_spec, hologram=None, pos=0):
        b, t, d = x_spec.shape
        
        # Proyecciones 100% Espectrales (0 FWHTs aquí)
        q_spec = self.q_filter(x_spec)
        k_spec = self.k_filter(x_spec)
        v_spec = self.v_filter(x_spec)
        
        # Saliencia en el espectro (Parseval)
        saliency = torch.sigmoid((x_spec * self.saliency_vec).sum(dim=-1, keepdim=True))
        
        # Roll para mezcla de secuencia (Requiere el índice de desplazamiento)
        idx = torch.arange(d, device=x_spec.device)
        shifts = (torch.arange(t, device=x_spec.device) + pos) % d
        shift_idx = (idx.unsqueeze(0) - shifts.unsqueeze(1)) % d
        
        # El Shift es la única operación que "mueve" información entre frecuencias
        k_shifted = torch.gather(k_spec, 2, shift_idx.unsqueeze(0).expand(b, -1, -1))
        
        # Acumulación Holográfica
        kv = (k_shifted * v_spec) * saliency
        h_acc = torch.cumsum(kv, dim=1)
        if hologram is not None:
            h_acc = h_acc + hologram.unsqueeze(1)
            
        # Recall
        h_norm = F.normalize(h_acc, p=2, dim=2, eps=1e-8)
        recall = q_spec * h_norm
        
        return self.o_filter(recall), h_acc[:, -1, :]

class OptimizedZeroGravityBlock(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.hra = SpectralHolographicAttention_v84(args)
        self.moe = ResonantSpectralMoE_v84(args)
        self.norm1 = SpectralRMSNorm(args.dim)
        self.norm2 = SpectralRMSNorm(args.dim)

    def forward(self, x_spec, hologram=None, pos=0):
        # TODO ESTO OCURRE SIN SALIR DEL DOMINIO DE WALSH
        h_attn, new_hologram = self.hra(self.norm1(x_spec), hologram, pos)
        x_spec = x_spec + h_attn
        x_spec = x_spec + self.moe(self.norm2(x_spec))
        return x_spec, new_hologram

class SpectralThinkerV8_4(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.args = args
        # Embedding: Entramos en espacial, pasamos a espectral una sola vez
        self.codes = nn.Parameter(torch.randn(args.vocab_size, args.emb_dim) * 0.02)
        self.basis = nn.Parameter(torch.randn(args.emb_dim, args.dim) * 0.02)
        
        self.layers = nn.ModuleList([OptimizedZeroGravityBlock(args) for _ in range(args.n_layers)])
        self.norm_final = SpectralRMSNorm(args.dim)

    def forward(self, tokens, targets=None, holograms=None, pos=0, use_cache=False):
        # 1. Entrada: Espacial -> Espectral (1 sola vez)
        z = F.embedding(tokens, self.codes)
        h_spatial = torch.matmul(z, self.basis)
        h_spec = fwht(h_spatial.view(-1, self.args.dim)).view(h_spatial.shape)
        
        # 2. Loop de Capas: 100% Espectral (0 FWHTs internos)
        new_holograms = []
        for i, layer in enumerate(self.layers):
            prev_h = holograms[i] if holograms is not None else None
            h_spec, new_h = layer(h_spec, prev_h, pos)
            new_holograms.append(new_h)
            
        # 3. Salida: Espectral -> Espacial (1 sola vez para el linear head)
        h_final_spec = self.norm_final(h_spec)
        h_final_spatial = fwht(h_final_spec.view(-1, self.args.dim)).view(h_final_spec.shape)
        
        # Proyección de Logits (Matrix-Free Factorizada)
        # Reutilizamos la lógica de v8.3 pero adaptada
        latent_h = torch.matmul(h_final_spatial, self.basis.t())
        logits = torch.matmul(latent_h, self.codes.t())
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
            
        return (logits, new_holograms) if use_cache else logits
