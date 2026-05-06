"""
model/model_analog_hybrid.py — TinyThinker con Placas Analógicas e Inyección Simbólica (DGE)

Evolución de model_analog.py que integra Diferenciabilidad Mixta.
Añade un 5º banco al AnalogFeedForward: el "Banco Simbólico".
Este banco usa operaciones no-derivables (Modulo, Floor) optimizadas mediante DGE.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class AnalogArgs:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 16384
    multiple_of: int = 256
    ffn_dim_multiplier: float = 2.0
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024

# --- STE (Straight-Through Estimator) ---
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_a, input_b, op_type):
        ctx.op_type = op_type
        if op_type == "mod":
            return torch.remainder(input_a, input_b)
        elif op_type == "div":
            return torch.floor(input_a / input_b)
        elif op_type == "round":
            return torch.round(input_a)
        else:
            return torch.sign(input_a)

    @staticmethod
    def backward(ctx, grad_output):
        # El gradiente pasa a ambos operandos (A y B)
        return grad_output, grad_output, None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SymbolicBank(nn.Module):
    """
    Banco de Operadores Prohibidos (No-Derivables).
    Sus parámetros deben ser optimizados mediante DGE.
    """
    def __init__(self, dim, bank_dim):
        super().__init__()
        # Proyecciones para obtener los operandos A y B
        # Inicializamos pequeño para estabilidad
        self.w_a = nn.Parameter(torch.randn(bank_dim, dim) * 0.02)
        self.w_b = nn.Parameter(torch.randn(bank_dim, dim) * 0.02)
        
        # Mezclador interno del banco (4 tipos de operaciones lógicas/aritméticas)
        # 0: Modulo, 1: Floor Div, 2: Round, 3: Sign/Step
        self.w_ops = nn.Parameter(torch.randn(bank_dim, 4) * 0.1)

        # Mezclador interno del banco (4 tipos de operaciones lógicas/aritméticas)
        self.w_ops = nn.Parameter(torch.randn(bank_dim, 4) * 0.1)
        self.norm = nn.LayerNorm(bank_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bsz, seq, dim)
        a = F.linear(x, self.w_a)
        b = F.linear(x, self.w_b)
        b_safe = torch.clamp(torch.abs(b), min=0.5) 
        
        # Usamos STE multivariable
        o_mod = STEFunction.apply(a, b_safe, "mod")
        o_div = STEFunction.apply(a, b_safe, "div")
        o_round = STEFunction.apply(a, b_safe, "round")
        o_step = STEFunction.apply(a, b_safe, "step")
        
        # Stack ops: (bsz, seq, bank_dim, 4)
        ops = torch.stack([o_mod, o_div, o_round, o_step], dim=-1)
        
        # Mezcla simbólica y Normalización (CRÍTICO para escala)
        out = torch.sum(ops * self.w_ops, dim=-1)
        return self.norm(out)

class AnalogFeedForwardHybrid(nn.Module):
    def __init__(self, args: AnalogArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        # Dividimos la carga de trabajo entre los 4 bancos analíticos
        self.bank_dim = hidden_dim // 4
        
        # 1. Banco Lineal
        self.w_linear = nn.Linear(args.dim, self.bank_dim, bias=False)
        
        # 2. Banco Multiplicativo
        self.group_size = 2
        self.w_mult = nn.Linear(args.dim, self.bank_dim * self.group_size, bias=False)
        
        # 3. Banco de Varianza
        self.w_var = nn.Linear(args.dim, self.bank_dim * self.group_size, bias=False)
        
        # 4. Banco Periódico
        self.w_sin = nn.Linear(args.dim, self.bank_dim, bias=False)
        
        # 5. BANCO SIMBÓLICO (Minoría DGE - Fijo a 8 neuronas para velocidad)
        self.sym_dim = 8
        self.symbolic = SymbolicBank(args.dim, self.sym_dim)
        
        # Mezclador final (Analítico) - Ahora suma bank_dim*4 + sym_dim
        self.w_out = nn.Linear(self.bank_dim * 4 + self.sym_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        
        # Analíticos (Adam)
        out_linear = F.silu(self.w_linear(x))
        
        mult_proj = self.w_mult(x).view(bsz, seq_len, self.bank_dim, self.group_size)
        out_mult = torch.tanh(torch.prod(mult_proj, dim=-1))
        
        var_proj = self.w_var(x).view(bsz, seq_len, self.bank_dim, self.group_size)
        out_var = torch.var(var_proj, dim=-1)
        
        out_sin = torch.sin(self.w_sin(x))
        
        # Simbólico (DGE)
        out_sym = self.symbolic(x)
        
        # Concatenar y proyectar (Analíticos + Simbólico)
        h_concat = torch.cat([out_linear, out_mult, out_var, out_sin, out_sym], dim=-1)
        return self.w_out(h_concat)

# --- RESTO DE LA ARQUITECTURA (Similitud con Llama/TinyThinker) ---

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    return torch.outer(t, freqs).float()

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def _reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs.view(*shape)

    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    freqs = _reshape_for_broadcast(freqs, xq_[..., 0])
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    xq_0, xq_1 = xq_.unbind(-1)
    xk_0, xk_1 = xk_.unbind(-1)
    xq_out = torch.stack([xq_0*cos - xq_1*sin, xq_0*sin + xq_1*cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk_0*cos - xk_1*sin, xk_0*sin + xk_1*cos], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: AnalogArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads else args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs=freqs_cis)
        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim).flatten(2, 3)
            xv = xv[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim).flatten(2, 3)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=True if mask is None else False)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)

class TransformerBlockHybrid(nn.Module):
    def __init__(self, args: AnalogArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = AnalogFeedForwardHybrid(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))

class TinyThinkerAnalogHybrid(nn.Module):
    def __init__(self, args: AnalogArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlockHybrid(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)

    def forward(self, tokens, targets=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(h.device)[:seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.zeros(seqlen, seqlen, device=tokens.device, dtype=h.dtype)
            mask.masked_fill_(torch.ones(seqlen, seqlen, device=tokens.device, dtype=torch.bool).tril().logical_not(), float('-inf'))
            mask = mask.view(1, 1, seqlen, seqlen)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        logits = self.output(self.norm(h))
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits
