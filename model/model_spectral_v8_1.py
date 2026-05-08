"""
model_spectral_v8_1.py — The Compressed Holographic MoE
Optimización extrema: Expertos Factorizados.

Reducción de parámetros: 1.6B -> ~180M (8x compresión de expertos).
Velocidad: ~4x más rápido en CPU que la V8.
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
    n_heads: int = 16
    n_kv_heads: int = 4
    vocab_size: int = 32768
    num_experts: int = 131072
    top_k: int = 16
    k_dim: int = 128         # Dimensión comprimida de los expertos
    max_batch_size: int = 32
    max_seq_len: int = 1024

# --- TRANSFORMADA RÁPIDA ---
try:
    from kernels.fwht_op import fwht_native
except ImportError:
    fwht_native = None

def fwht_iterative(x):
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

def fwht(x):
    orig_shape = x.shape
    if len(orig_shape) > 2:
        x_flat = x.reshape(-1, orig_shape[-1])
    else:
        x_flat = x
        
    if fwht_native is not None and x.device.type == 'cpu':
        res = fwht_native(x_flat)
        if res is not None:
            return res.view(orig_shape)
            
    return fwht_iterative(x_flat).view(orig_shape)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if hasattr(F, 'rms_norm'):
            return F.rms_norm(x, (x.shape[-1],), w, self.eps)
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * w

# --- MOE JERÁRQUICO COMPRIMIDO (Factorizado) ---
class CompressedSpectralMoE(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.dim = args.dim
        self.k_dim = args.k_dim 
        self.num_clans = 512  # Dividimos 131k en 512 clanes
        self.experts_per_clan = self.num_experts // self.num_clans
        
        # Banco de clanes (512 especialistas de alto nivel)
        self.clan_signatures = nn.Parameter(torch.randn(self.num_clans, self.k_dim) * 0.02)
        
        # Banco de expertos (Los 131k, agrupados)
        self.latent_signatures = nn.Parameter(torch.randn(args.num_experts, self.k_dim) * 0.02)
        self.latent_weights = nn.Parameter(torch.randn(args.num_experts, self.k_dim) * 0.01)
        
        # Base Espectral Compartida
        i = torch.arange(self.k_dim).view(self.k_dim, 1)
        j = torch.arange(self.dim).view(1, self.dim)
        basis = torch.cos(math.pi * i * (2 * j + 1) / (2 * self.dim))
        basis[0, :] *= 1.0 / math.sqrt(2.0)
        basis *= math.sqrt(2.0 / self.dim)
        self.register_buffer('basis', basis)

    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        x_spec = fwht(x_flat)
        x_latent = x_spec @ self.basis.t()
        
        # PASO 1: GATING DE CLANES (Mucho más pequeño y estable)
        clan_scores = torch.mm(x_latent, F.normalize(self.clan_signatures, p=2, dim=1, eps=1e-8).t())
        best_clan = torch.argmax(clan_scores, dim=1) # (BT)
        
        # PASO 2: GATING LOCAL VECTORIZADO (Sin bucles)
        # Reshapeamos para acceder por clan
        sigs_reshaped = self.latent_signatures.view(self.num_clans, self.experts_per_clan, self.k_dim)
        weights_reshaped = self.latent_weights.view(self.num_clans, self.experts_per_clan, self.k_dim)
        
        # Seleccionamos las firmas y pesos del clan ganador para cada token
        selected_sigs = sigs_reshaped[best_clan]       # (BT, experts_per_clan, k_dim)
        selected_weights = weights_reshaped[best_clan] # (BT, experts_per_clan, k_dim)
        
        # Normalizamos firmas locales
        selected_sigs = F.normalize(selected_sigs, p=2, dim=2, eps=1e-8)
        
        # BMM para scores locales: (BT, 256, k_dim) @ (BT, k_dim, 1) -> (BT, 256)
        local_scores = torch.bmm(selected_sigs, x_latent.unsqueeze(-1)).squeeze(-1)
        
        # Top-K local
        top_scores, top_indices = torch.topk(local_scores, k=min(self.top_k, self.experts_per_clan), dim=1)
        soft_weights = F.softmax(top_scores * 5.0, dim=1)
        
        # Obtenemos los pesos latentes finales
        idx = top_indices.unsqueeze(-1).expand(-1, -1, self.k_dim)
        chosen_latent_weights = torch.gather(selected_weights, 1, idx)
        
        out_latent = (chosen_latent_weights * soft_weights.unsqueeze(-1)).sum(dim=1)
        
        # Reconstrucción a dimensión completa
        out = out_latent @ self.basis
        
        return out.view(b, t, d)

# --- ATENCIÓN HOLOGRÁFICA ---
class HolographicAttention(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.dim = args.dim
        self.q_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.k_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.v_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.o_proj = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x, hologram=None, pos=0):
        b, t, d = x.shape
        # Proyecciones y FWHT inicial
        q = fwht(self.q_proj(x).view(-1, d)).view(b, t, d)
        k = fwht(self.k_proj(x).view(-1, d)).view(b, t, d)
        v = fwht(self.v_proj(x).view(-1, d)).view(b, t, d)
        
        # 1. Vectorizar desplazamientos (Shifts)
        # Creamos matriz de índices para desplazamiento circular paralelo
        idx = torch.arange(d, device=x.device)
        shifts = (torch.arange(t, device=x.device) + pos) % d
        # (t, d) índices de desplazamiento
        shift_idx = (idx.unsqueeze(0) - shifts.unsqueeze(1)) % d
        # Expandir para batch y aplicar gather: (b, t, d)
        k_shifted = torch.gather(k, 2, shift_idx.unsqueeze(0).expand(b, -1, -1))
        
        # 2. Generar memoria acumulada (Holograma)
        kv = k_shifted * v
        h_acc = torch.cumsum(kv, dim=1)
        
        if hologram is not None:
            h_acc = h_acc + hologram.unsqueeze(1)
            
        # 3. Recall con normalización (Vectorizado)
        # Normalizamos la memoria acumulada para estabilidad
        h_norm = F.normalize(h_acc, p=2, dim=2, eps=1e-8)
        recall = q * h_norm
        
        # Transformada final y proyección de salida
        out = fwht(recall.view(-1, d)).view(b, t, d)
        return self.o_proj(out), h_acc[:, -1, :]

# --- BLOQUE TRANSFORMER V8.1 ---
class SpectralV8_1Block(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.hra = HolographicAttention(args)
        self.moe = CompressedSpectralMoE(args)
        self.norm1 = RMSNorm(args.dim)
        self.norm2 = RMSNorm(args.dim)

    def forward(self, x, hologram=None, pos=0):
        h_attn, new_hologram = self.hra(self.norm1(x), hologram, pos)
        x = x + h_attn
        x = x + self.moe(self.norm2(x))
        return x, new_hologram

# --- EL PENSADOR V8.1 ---
class SpectralThinkerV8_1(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.emb_dim)
        self.emb_proj = nn.Linear(args.emb_dim, args.dim, bias=False)
        
        self.layers = nn.ModuleList([SpectralV8_1Block(args) for _ in range(args.n_layers)])
        self.norm_final = RMSNorm(args.dim)

        # Inicialización de estabilidad
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
