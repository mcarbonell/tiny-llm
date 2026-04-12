import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim: int = 512              
    n_layers: int = 8           
    n_heads: int = 8            
    n_kv_heads: int = 4         
    vocab_size: int = 16384     
    multiple_of: int = 256      
    ffn_dim_multiplier: float = 2.0 
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024     
    # MoE Args (Phase 1)
    n_experts: int = 8
    top_k: int = 2
    n_reserved: int = 4
    # COGA Scratchpad Args (Phase 2)
    n_scratch_slots: int = 32   
    # COGA Recurrence Args (Phase 4)
    n_pre_layers: int = 2       
    n_core_layers: int = 4      
    n_post_layers: int = 2      
    max_recurrence_steps: int = 4 
    # LoRA
    lora_r: int = 0             
    lora_alpha: float = 16.0     
    lora_dropout: float = 0.0    

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    return freqs

def reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs.shape == (x.shape[1], x.shape[-1]), f"{freqs.shape} vs {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    freqs = reshape_for_broadcast(freqs, xq_[..., 0])
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    xq_0, xq_1 = xq_.unbind(-1)
    xk_0, xk_1 = xk_.unbind(-1)
    xq_out_0 = xq_0 * cos - xq_1 * sin
    xq_out_1 = xq_0 * sin + xq_1 * cos
    xk_out_0 = xk_0 * cos - xk_1 * sin
    xk_out_1 = xk_0 * sin + xk_1 * cos
    xq_out = torch.stack([xq_out_0, xq_out_1], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_0, xk_out_1], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_experts = args.n_experts
        self.top_k = args.top_k
        self.n_reserved = args.n_reserved
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier)
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

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, past_key_value=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs=freqs_cis)
        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, seqlen, self.n_local_kv_heads, self.n_rep, self.head_dim).flatten(2, 3)
            xv = xv[:, :, :, None, :].expand(bsz, seqlen, self.n_local_kv_heads, self.n_rep, self.head_dim).flatten(2, 3)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            xk = torch.cat([past_key, xk], dim=2)
            xv = torch.cat([past_value, xv], dim=2)
            
        # OPTIMIZACIÓN: Scaled Dot Product Attention
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), (xk, xv)

class CrossAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, query: torch.Tensor, scratchpad: torch.Tensor):
        bsz, seqlen, _ = query.shape
        _, slots, _ = scratchpad.shape
        xq = self.wq(query).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(scratchpad).view(bsz, slots, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(scratchpad).view(bsz, slots, self.n_heads, self.head_dim).transpose(1, 2)
        
        # OPTIMIZACIÓN: Scaled Dot Product Attention (Sin máscara causal para el scratchpad)
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.cross_attention = CrossAttention(args)
        self.feed_forward = MoEFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, scratchpad: torch.Tensor, past_key_value=None, use_cache: bool = False, train_reserved: bool = False):
        attn_out, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask, past_key_value)
        h = x + attn_out
        cross_out = self.cross_attention(self.cross_attention_norm(h), scratchpad)
        h = h + cross_out
        ffn_out = self.feed_forward(self.ffn_norm(h), train_reserved=train_reserved)
        out = h + ffn_out
        if use_cache or past_key_value is not None:
            return out, new_kv
        return out

class TinyThinkerCOGA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.dim = args.dim
        self.n_scratch_slots = args.n_scratch_slots
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pre_layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_pre_layers)])
        self.core_layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_core_layers)])
        self.post_layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_post_layers)])
        self.halt_head = nn.Linear(args.dim, 1)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, tokens: torch.Tensor, scratchpad: Optional[torch.Tensor] = None, past_key_values=None, use_cache=False, train_reserved=False):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        if scratchpad is None:
            scratchpad = torch.zeros(bsz, self.n_scratch_slots, self.dim, device=tokens.device, dtype=h.dtype)
        past_len = 0
        if past_key_values:
            past_len = past_key_values[0][0].shape[2]
        freqs_cis = self.freqs_cis[past_len:past_len + seqlen]
        mask = None
        if seqlen > 1 and past_key_values is None:
            mask = torch.zeros(seqlen, seqlen, device=tokens.device, dtype=h.dtype)
            bool_mask = torch.ones(seqlen, seqlen, device=tokens.device, dtype=torch.bool).tril(diagonal=0).logical_not()
            mask = mask.masked_fill(bool_mask, float("-inf")).view(1, 1, seqlen, seqlen)
        past_key_values_out = []
        layer_idx = 0
        for layer in self.pre_layers:
            past_kv = past_key_values[layer_idx] if past_key_values else None
            layer_out = layer(h, freqs_cis, mask, scratchpad, past_kv, use_cache=use_cache, train_reserved=train_reserved)
            if isinstance(layer_out, tuple):
                h, past_kv_out = layer_out
                past_key_values_out.append(past_kv_out)
            else:
                h = layer_out
            layer_idx += 1
        halt_logits = self.halt_head(h[:, -1:, :])
        halt_prob = torch.sigmoid(halt_logits).squeeze(-1)
        steps_to_run = self.args.max_recurrence_steps
        if not self.training and bsz == 1:
            estimated_steps = max(1, round((1.0 - halt_prob.item()) * self.args.max_recurrence_steps))
            steps_to_run = estimated_steps
        for step in range(steps_to_run):
            core_layer_idx_start = layer_idx
            for layer in self.core_layers:
                is_last_step = (step == steps_to_run - 1)
                use_cache_here = use_cache and is_last_step
                past_kv = past_key_values[core_layer_idx_start] if past_key_values else None
                layer_out = layer(h, freqs_cis, mask, scratchpad, past_kv, use_cache=use_cache_here, train_reserved=train_reserved)
                if isinstance(layer_out, tuple):
                    h, past_kv_out = layer_out
                    if is_last_step:
                        past_key_values_out.append(past_kv_out)
                else:
                    h = layer_out
                core_layer_idx_start += 1
        layer_idx += self.args.n_core_layers
        for layer in self.post_layers:
            past_kv = past_key_values[layer_idx] if past_key_values else None
            layer_out = layer(h, freqs_cis, mask, scratchpad, past_kv, use_cache=use_cache, train_reserved=train_reserved)
            if isinstance(layer_out, tuple):
                h, past_kv_out = layer_out
                past_key_values_out.append(past_kv_out)
            else:
                h = layer_out
            layer_idx += 1
        h = self.norm(h)
        logits = self.output(h)
        if use_cache:
            return logits, past_key_values_out
        return logits
