"""
model_spectral_v8_6b_gated.py — The "Blueprint Stage-Gated" Architecture
Based on v8.6 Universal but adding learnable gates for Phase 1 pre-training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from model.model_spectral_v8_6_universal import SpectralThinkerV8_6, fwht_universal
from model.model_spectral_v8_4_optimized import SpectralRMSNorm, SpectralArgs

class GatedSpectralLinear(nn.Module):
    """
    Filtro Diagonal con Gating Multiplicativo.
    y = (x * weights) * gate
    En Fase 1: weights están congelados, solo gate aprende.
    """
    def __init__(self, dim):
        super().__init__()
        # Inicialización compatible con los pesos aprendidos de v8.6
        self.diag = nn.Parameter(torch.randn(dim) * 0.02)
        # El gate se inicializa a 1.0 para preservar el comportamiento original
        self.gate = nn.Parameter(torch.ones(dim))

    def forward(self, x_spec):
        return x_spec * self.diag * self.gate

class GatedResonantSpectralMoE(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.dim = args.dim
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.expert_signatures = nn.Parameter(torch.randn(args.num_experts, args.dim) * 0.02)
        self.expert_filters = nn.Parameter(torch.randn(args.num_experts, args.dim) * 0.02)
        # Gate compartido o por experto? El blueprint sugiere gating después de la proyección.
        # Para mantener la eficiencia de parámetros, usaremos un gate por salida de MoE.
        self.output_gate = nn.Parameter(torch.ones(args.dim))

    def forward(self, x_spec):
        b, t, d = x_spec.shape
        x_flat = x_spec.view(-1, d)
        
        logits = torch.mm(x_flat, F.normalize(self.expert_signatures, p=2, dim=1).t())
        scores, indices = torch.topk(logits, self.top_k, dim=1)
        weights = F.softmax(scores * 5.0, dim=1)
        
        selected_filters = self.expert_filters[indices]
        res_filter = (selected_filters * weights.unsqueeze(-1)).sum(dim=1)
        
        # Aplicamos pesos aprendidos * gate
        out_spec = (x_flat * res_filter) * self.output_gate
        return out_spec.view(b, t, d)

class GatedSpectralHolographicAttention(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.dim = args.dim
        # Sustituimos SpectralLinear por la versión Gated
        self.q_filter = GatedSpectralLinear(args.dim)
        self.k_filter = GatedSpectralLinear(args.dim)
        self.v_filter = GatedSpectralLinear(args.dim)
        self.o_filter = GatedSpectralLinear(args.dim)
        self.saliency_vec = nn.Parameter(torch.ones(args.dim))
        self.saliency_gate = nn.Parameter(torch.ones(1)) # Gate escalar para saliencia

    def forward(self, x_spec, hologram=None, pos=0):
        b, t, d = x_spec.shape
        
        q_spec = self.q_filter(x_spec)
        k_spec = self.k_filter(x_spec)
        v_spec = self.v_filter(x_spec)
        
        saliency = torch.sigmoid((x_spec * self.saliency_vec).sum(dim=-1, keepdim=True) * self.saliency_gate)
        
        idx = torch.arange(d, device=x_spec.device)
        shifts = (torch.arange(t, device=x_spec.device) + pos) % d
        shift_idx = (idx.unsqueeze(0) - shifts.unsqueeze(1)) % d
        
        k_shifted = torch.gather(k_spec, 2, shift_idx.unsqueeze(0).expand(b, -1, -1))
        
        kv = (k_shifted * v_spec) * saliency
        h_acc = torch.cumsum(kv, dim=1)
        if hologram is not None:
            h_acc = h_acc + hologram.unsqueeze(1)
            
        h_norm = F.normalize(h_acc, p=2, dim=2, eps=1e-8)
        recall = q_spec * h_norm
        
        return self.o_filter(recall), h_acc[:, -1, :]

class GatedOptimizedZeroGravityBlock(nn.Module):
    def __init__(self, args: SpectralArgs):
        super().__init__()
        self.hra = GatedSpectralHolographicAttention(args)
        self.moe = GatedResonantSpectralMoE(args)
        self.norm1 = SpectralRMSNorm(args.dim)
        self.norm2 = SpectralRMSNorm(args.dim)
        # Gates para los residuos (opcional pero potente para Phase 1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x_spec, hologram=None, pos=0):
        h_attn, new_hologram = self.hra(self.norm1(x_spec), hologram, pos)
        x_spec = x_spec + h_attn * self.alpha
        
        h_moe = self.moe(self.norm2(x_spec))
        x_spec = x_spec + h_moe * self.beta
        
        return x_spec, new_hologram

class SpectralThinkerV8_6b(SpectralThinkerV8_6):
    """
    Implementación del Stage-Gating. 
    Añade gates a cada proyección y bloque residual.
    """
    def __init__(self, args: SpectralArgs):
        # Inicialización base (creará self.codes, self.basis, etc.)
        super().__init__(args)
        
        # Sobreescribimos las capas con la versión Gated
        self.layers = nn.ModuleList([GatedOptimizedZeroGravityBlock(args) for _ in range(args.n_layers)])
        
        # Gates para embeddings y salida final
        self.emb_gate = nn.Parameter(torch.ones(args.dim))
        self.output_gate = nn.Parameter(torch.ones(args.vocab_size))

    def forward(self, tokens, targets=None, holograms=None, pos=0, use_cache=False):
        device = self.codes.device
        if tokens.device != device:
            tokens = tokens.to(device)
            
        # 1. Entrada con Gate
        z = F.embedding(tokens, self.codes)
        h_spatial = torch.matmul(z, self.basis)
        h_spec = fwht_universal(h_spatial) * self.emb_gate
        
        # 2. Loop de Capas Gated
        new_holograms = []
        for i, layer in enumerate(self.layers):
            prev_h = holograms[i] if holograms is not None else None
            if getattr(layer, 'use_checkpoint', False) and self.training:
                h_spec, new_h = torch.utils.checkpoint.checkpoint(layer, h_spec, prev_h, pos, use_reentrant=False)
            else:
                h_spec, new_h = layer(h_spec, prev_h, pos)
            new_holograms.append(new_h)
            
        # 3. Salida con Gate
        h_final_spec = self.norm_final(h_spec)
        h_final_spatial = fwht_universal(h_final_spec)
        
        latent_h = torch.matmul(h_final_spatial, self.basis.t())
        logits = torch.matmul(latent_h, self.codes.t()) * self.output_gate
        
        if targets is not None:
            if targets.device != device:
                targets = targets.to(device)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
            
        return (logits, new_holograms) if use_cache else logits
