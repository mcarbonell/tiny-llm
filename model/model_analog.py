"""
model_analog.py — TinyThinker con Neuronas Analógicas (EXP-9)

Esta arquitectura reemplaza el Feed-Forward Network (MLP) estándar de un
Transformer por una "Placa de Circuitos Evolutiva" (AnalogFeedForward).
En lugar de depender exclusivamente de sumas ponderadas seguidas de activaciones
no lineales genéricas (SiLU/GELU), divide la dimensión oculta en cuatro bancos
matemáticos especializados que operan en paralelo:

1. Banco Lineal (SUM): y = silu(x @ W) -> Procesamiento semántico/asociativo.
2. Banco Multiplicativo (PROD): y = tanh(prod(x @ W)) -> Lógica AND/condicional estricta.
3. Banco de Varianza (VAR): y = var(x @ W) -> Detección de anomalías/contrastes.
4. Banco Periódico (SIN): y = sin(x @ W) -> Resolución de alta frecuencia / XOR.

Al concatenar las respuestas de los 4 bancos, la red puede resolver
problemas lógicos complejos en O(1) profundidad que a un MLP clásico le
costarían docenas de capas aproximar.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
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

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == 'privateuseone':
            with torch.autocast('privateuseone', enabled=False):
                x_f = x.float()
                var = (x_f * x_f).mean(-1, keepdim=True)
                return (x_f * torch.rsqrt(var + self.eps)) * self.weight
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    return torch.outer(t, freqs).float()

def _reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs.shape == (x.shape[1], x.shape[-1]), f"{freqs.shape} vs {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        assert args.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis, mask, past_key_value=None):
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

        if past_key_value is not None:
            pk, pv = past_key_value
            xk = torch.cat([pk, xk], dim=2)
            xv = torch.cat([pv, xv], dim=2)

        if past_key_value is None and seqlen > 1:
            out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out), (xk, xv)

class AnalogFeedForward(nn.Module):
    """
    Placa de Circuitos Lógicos. Divide la computación en 4 bancos paralelos
    con diferentes propiedades matemáticas fundamentales.
    """
    def __init__(self, args: AnalogArgs):
        super().__init__()
        # Calculamos la dimensión oculta total como en un MLP clásico
        hidden_dim = int(2 * (4 * args.dim) / 3)
        if args.ffn_dim_multiplier:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        # Dividimos la carga de trabajo equitativamente entre los 4 bancos
        self.bank_dim = hidden_dim // 4
        
        # 1. Banco Lineal (Suma Ponderada Clásica) -> Semántica
        self.w_linear = nn.Linear(args.dim, self.bank_dim, bias=False)
        
        # 2. Banco Multiplicativo (Producto Ponderado) -> Lógica AND estricta
        # Usamos proyecciones en grupos pequeños (ej. pares o tríos) para evitar explosiones
        self.group_size = 2
        self.w_mult = nn.Linear(args.dim, self.bank_dim * self.group_size, bias=False)
        
        # 3. Banco de Varianza (Contraste Estadístico) -> Detector de Anomalías
        self.w_var = nn.Linear(args.dim, self.bank_dim * self.group_size, bias=False)
        
        # 4. Banco Periódico (Senoidal) -> Resolución de alta frecuencia / XOR
        self.w_sin = nn.Linear(args.dim, self.bank_dim, bias=False)
        
        # Mezclador de salida que combina los 4 bancos de vuelta a la dimensión del modelo
        self.w_out = nn.Linear(self.bank_dim * 4, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        
        # 1. Banco Lineal (Activación SiLU clásica)
        out_linear = F.silu(self.w_linear(x))
        
        # 2. Banco Multiplicativo
        # Proyectamos a un espacio más grande y multiplicamos en grupos
        mult_proj = self.w_mult(x).view(bsz, seq_len, self.bank_dim, self.group_size)
        # Prod a lo largo de la dimensión del grupo y Tanh para estabilizar
        out_mult = torch.tanh(torch.prod(mult_proj, dim=-1))
        
        # 3. Banco de Varianza
        # Proyectamos y medimos la varianza (contraste) dentro de cada grupo
        var_proj = self.w_var(x).view(bsz, seq_len, self.bank_dim, self.group_size)
        out_var = torch.var(var_proj, dim=-1)
        
        # 4. Banco Periódico
        # Las ondas senoidales dividen el espacio latente periódicamente
        out_sin = torch.sin(self.w_sin(x))
        
        # Concatenar los 4 "sabores" matemáticos (bsz, seq_len, bank_dim * 4)
        h_concat = torch.cat([out_linear, out_mult, out_var, out_sin], dim=-1)
        
        # Mezclar y proyectar de vuelta
        return self.w_out(h_concat)

class TransformerBlock(nn.Module):
    def __init__(self, args: AnalogArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = AnalogFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_checkpoint = False

    def _fwd(self, x, freqs_cis, mask, past_kv):
        attn_out, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask, past_kv)
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h)), new_kv

    def forward(self, x, freqs_cis, mask, past_key_value=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._fwd, x, freqs_cis, mask, past_key_value, use_reentrant=False)
        return self._fwd(x, freqs_cis, mask, past_key_value)

class TinyThinkerAnalog(nn.Module):
    def __init__(self, args: AnalogArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Weight tying desactivado para Analog también (mejor convergencia)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / math.sqrt(args.dim))

        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)

    def forward(self, tokens, targets=None, past_key_values=None, use_cache=False):
        _bsz, seqlen = tokens.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if tokens.device.type == 'privateuseone':
            with torch.autocast('privateuseone', enabled=False):
                h = self.tok_embeddings(tokens)
                freqs_cis = self.freqs_cis.to(h.device)[past_len:past_len + seqlen]
                mask = _causal_mask(seqlen, tokens.device, h.dtype) if past_key_values is None else None
        else:
            h = self.tok_embeddings(tokens)
            freqs_cis = self.freqs_cis.to(h.device)[past_len:past_len + seqlen]
            mask = _causal_mask(seqlen, tokens.device, h.dtype) if past_key_values is None else None

        new_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            h, kv = layer(h, freqs_cis, mask, pkv)
            new_kvs.append(kv)

        logits = self.output(self.norm(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        if use_cache:
            return logits, new_kvs
        return logits

def _causal_mask(seqlen, device, dtype):
    if seqlen <= 1: return None
    mask = torch.zeros(seqlen, seqlen, device=device, dtype=dtype)
    mask.masked_fill_(torch.ones(seqlen, seqlen, device=device, dtype=torch.bool).tril().logical_not(), float('-inf'))
    return mask.view(1, 1, seqlen, seqlen)

def count_params(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'trainable': trainable, 'trainable_M': round(trainable / 1e6, 2)}
