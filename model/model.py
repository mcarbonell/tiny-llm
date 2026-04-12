import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim: int = 512              # Dimensión del modelo (d_model)
    n_layers: int = 8           # Número de capas del Transformer
    n_heads: int = 8            # Número de "queries" en la atención
    n_kv_heads: int = 4         # GQA: Menos heads de K/V reduce la memoria dramáticamente
    vocab_size: int = 16384     # Nuestro vocabulario del BPE Tokenizer
    multiple_of: int = 256      # Usado en la capa neuronal (SwiGLU)
    ffn_dim_multiplier: float = 2.0 
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024     # ContextWindow base
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

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float, lora_r: int = 0, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = LoRALinear(dim, hidden_dim, bias=False, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.w2 = LoRALinear(hidden_dim, dim, bias=False, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

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

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            xk = torch.cat([past_key, xk], dim=2)
            xv = torch.cat([past_value, xv], dim=2)

        # OPTIMIZACIÓN: Scaled Dot Product Attention (Flash Attention si disponible)
        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=mask, 
            dropout_p=0.0, 
            is_causal=False 
        )
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), (xk, xv)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_checkpoint = False

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, past_key_value=None):
        def _forward_block(x, freqs_cis, mask, past_key_value):
            attn_out, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask, past_key_value)
            h = x + attn_out
            out = h + self.feed_forward(self.ffn_norm(h))
            return out, new_kv

        if self.use_checkpoint and self.training:
            # Recomputamos las activaciones durante el backward para ahorrar memoria
            # Nota: past_key_value debe ser None para que el checkpointing sea sencillo
            return torch.utils.checkpoint.checkpoint(_forward_block, x, freqs_cis, mask, past_key_value, use_reentrant=False)
        
        return _forward_block(x, freqs_cis, mask, past_key_value)

class TinyThinker(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, past_key_values=None, use_cache: bool = False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.view(1, 1, seqlen, seqlen)

        new_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            h, kv = layer(h, freqs_cis, mask, pkv)
            new_kvs.append(kv)

        h = self.norm(h)
        logits = self.output(h)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        if use_cache:
            return logits, new_kvs
        return logits
