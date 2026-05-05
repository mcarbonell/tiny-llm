"""
model_spectral_v8_3_matrix_free.py — The "Zero Gravity" Architecture
100% Matrix-Free: No N^2 dense matrices. All O(N log N) or O(N).

Este modelo rompe con la dependencia de las multiplicaciones de matrices densas.
Utiliza Filtros Espectrales Diagonales y Resonancia Holográfica.
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
    emb_dim: int = 128      # Dimensión latente del embedding (el "código JPEG")
    n_layers: int = 16
    vocab_size: int = 32768
    num_experts: int = 128  # Pocos expertos pero mucha resolución (D)
    top_k: int = 8          # Top-K reducido para D tan grande
    max_batch_size: int = 32
    max_seq_len: int = 4096

# --- TRANSFORMADA RÁPIDA (El Motor del Mezclado) ---
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

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight

# --- CAPA LINEAL MATRIX-FREE (Diagonal Spectral Filter) ---
class MatrixFreeLinear(nn.Module):
    """
    Reemplaza nn.Linear(D, D) por un Filtro Diagonal en el dominio de Walsh.
    Complejidad: O(D log D) vs O(D^2)
    Parámetros: D vs D^2
    """
    def __init__(self, dim):
        super().__init__()
        self.diag = nn.Parameter(torch.randn(dim) * 0.02)
        
    def forward(self, x):
        # x: (B, T, D) o (B, D)
        shape = x.shape
        x_flat = x.view(-1, shape[-1])
        # 1. Pasar al dominio de Walsh (Mezclado de información)
        x_spec = fwht(x_flat)
        # 2. Filtrar (Aprendizaje de características)
        x_filtered = x_spec * self.diag
        # 3. Volver al dominio espacial
        out = fwht(x_filtered)
        return out.view(*shape)

# --- MOE DE RESONANCIA ESPECTRAL (Matrix-Free) ---
class ResonantSpectralMoE(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.dim = args.dim
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        
        # Cada experto es solo una "Firma Espectral" (Vector D)
        self.expert_signatures = nn.Parameter(torch.randn(args.num_experts, args.dim) * 0.02)
        # Y un filtro de respuesta (Vector D)
        self.expert_filters = nn.Parameter(torch.randn(args.num_experts, args.dim) * 0.02)

    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        x_spec = fwht(x_flat) # Trabajamos en el dominio de Walsh
        
        # GATING POR RESONANCIA (Producto escalar paralelo)
        # (BT, D) @ (D, NumExperts) -> (BT, NumExperts)
        # Nota:mm aquí es O(D * NumExperts), pero NumExperts es externo. 
        # La relación con D es lineal.
        logits = torch.mm(x_spec, F.normalize(self.expert_signatures, p=2, dim=1).t())
        
        scores, indices = torch.topk(logits, self.top_k, dim=1)
        weights = F.softmax(scores * 5.0, dim=1)
        
        # Aplicación de Filtros Expertos (Matrix-Free)
        # En lugar de matrices, los expertos aplican su "filtro" al espectro
        out_spec = torch.zeros_like(x_spec)
        
        # Vectorizamos la aplicación de filtros
        for k in range(self.top_k):
            idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            # Seleccionamos el filtro del experto
            filters = self.expert_filters[idx]
            out_spec += w * (x_spec * filters)
            
        return fwht(out_spec).view(b, t, d)

# --- ATENCIÓN HOLOGRÁFICA MATRIX-FREE ---
class MatrixFreeHolographicAttention(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.dim = args.dim
        # Proyecciones Diagonales (Matrix-Free)
        self.q_filter = MatrixFreeLinear(args.dim)
        self.k_filter = MatrixFreeLinear(args.dim)
        self.v_filter = MatrixFreeLinear(args.dim)
        self.o_filter = MatrixFreeLinear(args.dim)
        
        # Saliency Gater (También Matrix-Free, usando un vector)
        self.saliency_vec = nn.Parameter(torch.ones(args.dim))

    def forward(self, x, hologram=None, pos=0):
        b, t, d = x.shape
        
        # Q, K, V generados sin matrices densas
        q = self.q_filter(x)
        k = self.k_filter(x)
        v = self.v_filter(x)
        
        # Saliencia (Gater Matrix-Free)
        # Usamos una operación de resonancia local para la saliencia
        saliency = torch.sigmoid((x * self.saliency_vec).sum(dim=-1, keepdim=True))
        
        # Lógica de Roll (Vectorizada como en v8.1)
        idx = torch.arange(d, device=x.device)
        shifts = (torch.arange(t, device=x.device) + pos) % d
        shift_idx = (idx.unsqueeze(0) - shifts.unsqueeze(1)) % d
        k_shifted = torch.gather(fwht(k.view(-1, d)).view(b, t, d), 2, shift_idx.unsqueeze(0).expand(b, -1, -1))
        
        # Acumulación Holográfica
        v_spec = fwht(v.view(-1, d)).view(b, t, d)
        kv = (k_shifted * v_spec) * saliency
        h_acc = torch.cumsum(kv, dim=1)
        
        if hologram is not None:
            h_acc = h_acc + hologram.unsqueeze(1)
            
        # Recall
        h_norm = F.normalize(h_acc, p=2, dim=2, eps=1e-8)
        q_spec = fwht(q.view(-1, d)).view(b, t, d)
        recall = q_spec * h_norm
        
        # Salida Matrix-Free
        out = fwht(recall.view(-1, d)).view(b, t, d)
        return self.o_filter(out), h_acc[:, -1, :]

# --- EMBEDDING ESPECTRAL COMPRIMIDO (Factorización Matrix-Free) ---
class SpectralEmbedding(nn.Module):
    """
    Comprime el vocabulario estilo JPEG. 
    En lugar de V x D, usa (V x k) + (k x D).
    """
    def __init__(self, vocab_size, dim, k):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.k = k
        
        # El "Código JPEG" de cada palabra
        self.codes = nn.Parameter(torch.randn(vocab_size, k) * 0.02)
        # La "Base Espectral" compartida
        self.basis = nn.Parameter(torch.randn(k, dim) * 0.02)

    def forward(self, tokens):
        # tokens: (B, T)
        # 1. Obtener los códigos latentes: (B, T, k)
        z = F.embedding(tokens, self.codes)
        # 2. Descomprimir al vuelo: (B, T, D)
        return torch.matmul(z, self.basis)

    def project_logits(self, h):
        # h: (B, T, D) -> logits: (B, T, V)
        # Operación factorizada: (h @ basis^T) @ codes^T
        # Mucho más rápido que h @ (basis^T @ codes^T)
        latent_h = torch.matmul(h, self.basis.t()) # (B, T, k)
        logits = torch.matmul(latent_h, self.codes.t()) # (B, T, V)
        return logits

class ZeroGravityBlock(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.hra = MatrixFreeHolographicAttention(args)
        self.moe = ResonantSpectralMoE(args)
        self.norm1 = RMSNorm(args.dim)
        self.norm2 = RMSNorm(args.dim)

    def forward(self, x, hologram=None, pos=0):
        h_attn, new_hologram = self.hra(self.norm1(x), hologram, pos)
        x = x + h_attn
        x = x + self.moe(self.norm2(x))
        return x, new_hologram

class SpectralThinkerV8_3(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.args = args
        # Reemplazamos Embedding denso por Espectral
        self.spectral_emb = SpectralEmbedding(args.vocab_size, args.dim, args.emb_dim)
        
        self.layers = nn.ModuleList([ZeroGravityBlock(args) for _ in range(args.n_layers)])
        self.norm_final = RMSNorm(args.dim)

    def forward(self, tokens, targets=None, holograms=None, pos=0, use_cache=False):
        # Entrada descomprimida al vuelo
        h = self.spectral_emb(tokens)
        
        new_holograms = []
        for i, layer in enumerate(self.layers):
            prev_h = holograms[i] if holograms is not None else None
            h, new_h = layer(h, prev_h, pos)
            new_holograms.append(new_h)
            
        # Salida factorizada (ultra-rápida)
        logits = self.spectral_emb.project_logits(self.norm_final(h))
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
            
        return (logits, new_holograms) if use_cache else logits
