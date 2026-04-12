import math
from dataclasses import dataclass
from typing import Optional, Tuple
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
    # MoE Args
    n_experts: int = 8          # Total de expertos
    top_k: int = 2              # Expertos activos por token
    n_reserved: int = 4         # Slots reservados (COGA Phase 1)
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

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, r: int = 0, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        if r > 0:
            self.lora_down = nn.Linear(in_features, r, bias=False)
            self.lora_up = nn.Linear(r, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        if self.r > 0:
            nn.init.zeros_(self.lora_up.weight)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = self.lora_up(self.lora_down(self.dropout(x)))
            return result + lora_out * self.scaling
        return result

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

        self.wq = LoRALinear(args.dim, args.n_heads * self.head_dim, bias=False, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        self.wk = LoRALinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        self.wv = LoRALinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
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

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MoEFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_checkpoint = False

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, past_key_value=None, use_cache: bool = False, train_reserved: bool = False):
        attn_out, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask, past_key_value)
        h = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(h), train_reserved=train_reserved)
        out = h + ffn_out
        if use_cache or past_key_value is not None:
            return out, new_kv
        return out

class TinyThinkerMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, tokens: torch.Tensor, past_key_values=None, use_cache=False, train_reserved=False):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
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
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            layer_out = layer(h, freqs_cis, mask, past_kv, use_cache=use_cache, train_reserved=train_reserved)
            if isinstance(layer_out, tuple):
                h, past_kv_out = layer_out
                past_key_values_out.append(past_kv_out)
            else:
                h = layer_out
        h = self.norm(h)
        logits = self.output(h)
        if use_cache:
            return logits, past_key_values_out
        return logits
