import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class CogaSpectralArgs:
    dim: int = 1024
    emb_dim: int = 128
    n_pre_layers: int = 2
    n_core_layers: int = 4
    n_post_layers: int = 2
    max_steps: int = 8          # Máximas iteraciones en el core
    n_heads: int = 16
    n_kv_heads: int = 4
    vocab_size: int = 32768
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    
    # Scratchpad
    scratchpad_slots: int = 64
    
    # Spectral Compression
    k_dim_attn: int = 128
    k_dim_ffn: int = 128
    k_hidden_ffn: int = 256
    k_seq_len: int = 64
    
    # MoE
    n_experts: int = 8
    top_k: int = 2

# --- Reutilización de capas de model_spectral_moe ---
from model.model_spectral_moe import DCTLinear, WalshLinear, RMSNorm, SpectralAttention, apply_rotary_emb, precompute_freqs_cis

class SpectralMoEExpert(nn.Module):
    def __init__(self, args: CogaSpectralArgs):
        super().__init__()
        hidden = int(2 * (4 * args.dim) / 3)
        hidden = args.multiple_of * ((hidden + args.multiple_of - 1) // args.multiple_of)
        self.w1 = WalshLinear(args.dim, hidden, args.k_dim_ffn, args.k_hidden_ffn)
        self.w2 = WalshLinear(hidden, args.dim, args.k_hidden_ffn, args.k_dim_ffn)
        self.w3 = WalshLinear(args.dim, hidden, args.k_dim_ffn, args.k_hidden_ffn)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SpectralMoEFeedForward(nn.Module):
    def __init__(self, args: CogaSpectralArgs):
        super().__init__()
        self.experts = nn.ModuleList([SpectralMoEExpert(args) for _ in range(args.n_experts)])
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        self.top_k = args.top_k
    def forward(self, x):
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        gate_logits = self.gate(x_flat)
        weights = F.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_indices, k_indices = (top_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                out[token_indices] += top_weights[token_indices, k_indices].unsqueeze(-1) * expert(x_flat[token_indices])
        return out.view(bsz, seqlen, dim)

# =============================================================================
# COGA Core Components
# =============================================================================

class Scratchpad(nn.Module):
    """
    Working Memory Bank (KV slots).
    Permite al modelo leer y escribir estados ocultos sin afectar el contexto autoregresivo.
    """
    def __init__(self, args: CogaSpectralArgs):
        super().__init__()
        self.slots = nn.Parameter(torch.zeros(args.scratchpad_slots, args.dim))
        nn.init.normal_(self.slots, std=0.02)
        
    def forward(self, x):
        # x: (bsz, seqlen, dim)
        # Retorna el banco de memoria expandido para el batch
        return self.slots.unsqueeze(0).expand(x.size(0), -1, -1)

class CogaCoreBlock(nn.Module):
    def __init__(self, args: CogaSpectralArgs):
        super().__init__()
        self.attention = SpectralAttention(args)
        self.feed_forward = SpectralMoEFeedForward(args)
        self.norm1 = RMSNorm(args.dim, eps=args.norm_eps)
        self.norm2 = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Cross-Attention al Scratchpad
        self.cross_attn = SpectralAttention(args) # Reutilizamos SpectralAttention para cross
        self.norm_cross = RMSNorm(args.dim, eps=args.norm_eps)
        
    def forward(self, x, freqs_cis, mask, memory):
        # 1. Auto-Atención
        h_attn, _ = self.attention(self.norm1(x), freqs_cis, mask)
        h = x + h_attn
        
        # 2. Cross-Atención al Scratchpad (Working Memory)
        h_mem, _ = self.cross_attn(self.norm_cross(h), None, None, past_kv=None) # Simplificado
        h = h + h_mem
        
        # 3. FFN (MoE)
        return h + self.feed_forward(self.norm2(h))

# =============================================================================
# The Spectral COGA Model (V2 Prototype)
# =============================================================================

class TinyThinkerCogaSpectralV2(nn.Module):
    def __init__(self, args: CogaSpectralArgs):
        super().__init__()
        self.args = args
        
        # Input
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.emb_dim)
        self.emb_proj = nn.Linear(args.emb_dim, args.dim, bias=False)
        
        # Phase I: Pre-Block (Parsing)
        from model.model_spectral_moe import SpectralTransformerBlock
        self.pre_layers = nn.ModuleList([SpectralTransformerBlock(args) for _ in range(args.n_pre_layers)])
        
        # Phase II: The Cerebellum (Halt Head)
        self.halt_head = nn.Linear(args.dim, 1)
        
        # Phase III: Core Block (Recurrent)
        self.core_block = CogaCoreBlock(args)
        self.scratchpad = Scratchpad(args)
        
        # Phase IV: Post-Block (Refinement)
        self.post_layers = nn.ModuleList([SpectralTransformerBlock(args) for _ in range(args.n_post_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Utils
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)
        
        # Init
        nn.init.normal_(self.tok_embeddings.weight, std=0.02)
        nn.init.normal_(self.emb_proj.weight, std=0.02)
        nn.init.zeros_(self.halt_head.bias)

    def forward(self, tokens, targets=None):
        bsz, seqlen = tokens.shape
        h = self.emb_proj(self.tok_embeddings(tokens))
        freqs_cis = self.freqs_cis.to(h.device)[:seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).view(1, 1, seqlen, seqlen)

        # 1. Pre-Processing
        for layer in self.pre_layers:
            h, _ = layer(h, freqs_cis, mask)

        # 2. Dynamic Thinking (Recurrence)
        # Halt gating
        halt_prob = torch.sigmoid(self.halt_head(h.mean(dim=1))) # (bsz, 1)
        num_steps = int(halt_prob.mean().item() * self.args.max_steps)
        num_steps = max(1, num_steps) # Al menos una pasada
        
        memory = self.scratchpad(h)
        for _ in range(num_steps):
            h = self.core_block(h, freqs_cis, mask, memory)

        # 3. Post-Processing
        for layer in self.post_layers:
            h, _ = layer(h, freqs_cis, mask)

        h_norm = self.norm(h)
        h_small = F.linear(h_norm, self.emb_proj.weight.t()) 
        logits = F.linear(h_small, self.tok_embeddings.weight)
        
        if targets is not None:
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits
