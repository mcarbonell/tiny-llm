import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from kernels.fused_residual_norm_op import fused_residual_norm
from model.walsh_linear_fwht import WalshLinearFWHT
from model.model_spectral_v10_hippocampus import WalshLinear


@dataclass
class UnifiedArgs:
    dim: int = 2048
    emb_dim: int = 0            # 0 = embedding directo (sin factorizar)
    n_layers: int = 8
    vocab_size: int = 32768
    max_seq_len: int = 1024
    k_walsh: int = 256
    use_hippocampus: bool = True
    k_mem: int = 32
    chunk_size: int = 256
    gamma: float = 0.9
    lambda_phase: float = 0.01
    spherical_head: bool = True
    weight_tying: bool = True
    use_fwht_kernel: bool = True   # False = matriz Walsh densa (ablation E)
    alpha_init: float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# nGPT / SPHERICAL UTILS
# ─────────────────────────────────────────────────────────────────────────────
def norm_sphere(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class SphericalHead(nn.Module):
    def __init__(self, in_features, out_features, init_tau=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return F.linear(x_norm, w_norm) * self.tau


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL COMPLEX FFT MIXER  (+ Hippocampus opcional)
# ─────────────────────────────────────────────────────────────────────────────
class CausalFFTBlock(nn.Module):
    def __init__(self, T, D, k_walsh, args: UnifiedArgs):
        super().__init__()
        self.T = T
        self.pad_T = 1
        while self.pad_T < 2 * T:
            self.pad_T *= 2
        self.n_freq = self.pad_T // 2 + 1

        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase = nn.Parameter(torch.zeros(self.n_freq))

        mask = torch.zeros(self.pad_T)
        mask[:T] = 1.0
        self.register_buffer('causal_mask', mask)

        self.out_proj = WalshLinearFWHT(D, D, k_walsh, normalized=True)

        self.use_hippocampus = args.use_hippocampus
        if self.use_hippocampus:
            self.k_mem = min(args.k_mem, self.n_freq)
            self.gamma = args.gamma
            self.read_gate = nn.Parameter(torch.ones(self.k_mem, 1, dtype=torch.complex64))
            self.write_gate = nn.Parameter(torch.ones(self.k_mem, 1, dtype=torch.complex64))

    def forward(self, x, memory_state=None):
        B, T, D = x.shape
        xt = x.permute(0, 2, 1)
        pad = torch.zeros(B, D, self.pad_T - T, device=x.device, dtype=x.dtype)
        xt_pad = torch.cat([xt, pad], dim=-1)

        X = torch.fft.rfft(xt_pad, dim=-1)

        gate_raw = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        h_raw = torch.fft.irfft(gate_raw, n=self.pad_T)
        h_causal = h_raw * self.causal_mask
        gate_causal = torch.fft.rfft(h_causal, n=self.pad_T)

        X_gated = X * gate_causal
        X_gated_perm = X_gated.permute(0, 2, 1)  # B, freq, D

        if self.use_hippocampus and memory_state is not None:
            X_gated_perm[:, :self.k_mem, :] = X_gated_perm[:, :self.k_mem, :] + memory_state * self.read_gate

        out = torch.fft.irfft(X_gated_perm.permute(0, 2, 1), n=self.pad_T, dim=-1)[..., :T]
        out = out.permute(0, 2, 1)

        if self.use_hippocampus:
            current_mem = X_gated_perm[:, :self.k_mem, :].clone()
            if memory_state is None:
                new_memory_state = current_mem * self.write_gate
            else:
                new_memory_state = self.gamma * memory_state + (1 - self.gamma) * (current_mem * self.write_gate)
        else:
            new_memory_state = None

        return self.out_proj(out), new_memory_state

    def get_phase_loss(self):
        diffs = self.phase[1:] - self.phase[:-1]
        return torch.mean(torch.abs(diffs))


class MatrixFreeFFN(nn.Module):
    def __init__(self, D, k_walsh, use_fwht_kernel=True):
        super().__init__()
        if use_fwht_kernel:
            self.proj = WalshLinearFWHT(D, D, k_walsh, normalized=True)
        else:
            self.proj = WalshLinear(D, D, k_walsh, normalized=True)

    def forward(self, x):
        return F.gelu(self.proj(x))


class nGPTBlock(nn.Module):
    def __init__(self, args: UnifiedArgs):
        super().__init__()
        self.mixer = CausalFFTBlock(args.max_seq_len, args.dim, args.k_walsh, args)
        self.ffn = MatrixFreeFFN(args.dim, args.k_walsh, args.use_fwht_kernel)
        self.alpha_m = nn.Parameter(torch.full((args.dim,), args.alpha_init))
        self.alpha_f = nn.Parameter(torch.full((args.dim,), args.alpha_init))

    def forward(self, x, state=None):
        m, next_state = self.mixer(x, state)
        m = norm_sphere(m)
        x = fused_residual_norm(x, m, self.alpha_m)
        f = norm_sphere(self.ffn(x))
        x = fused_residual_norm(x, f, self.alpha_f)
        return x, next_state


# ─────────────────────────────────────────────────────────────────────────────
# MODELO UNIFICADO
# ─────────────────────────────────────────────────────────────────────────────
class UnifiedSpectral(nn.Module):
    def __init__(self, args: UnifiedArgs):
        super().__init__()
        self.args = args

        if args.emb_dim and args.emb_dim > 0:
            self.embed = nn.Embedding(args.vocab_size, args.emb_dim)
            self.embed_proj = nn.Linear(args.emb_dim, args.dim, bias=False)
            self.use_factorized = True
        else:
            self.embed = nn.Embedding(args.vocab_size, args.dim)
            self.embed_proj = None
            self.use_factorized = False

        self.blocks = nn.ModuleList([nGPTBlock(args) for _ in range(args.n_layers)])

        if args.spherical_head:
            if self.use_factorized:
                self.head_proj = nn.Linear(args.dim, args.emb_dim, bias=False)
                self.head = SphericalHead(args.emb_dim, args.vocab_size, init_tau=10.0)
                if args.weight_tying:
                    self.head.weight = self.embed.weight
            else:
                self.head_proj = None
                self.head = SphericalHead(args.dim, args.vocab_size, init_tau=10.0)
                if args.weight_tying:
                    self.head.weight = self.embed.weight
        else:
            if self.use_factorized:
                self.head_proj = nn.Linear(args.dim, args.emb_dim, bias=False)
                self.head = nn.Linear(args.emb_dim, args.vocab_size, bias=False)
                if args.weight_tying:
                    self.head.weight = self.embed.weight
            else:
                self.head_proj = None
                self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x_full):
        if self.use_factorized:
            e = norm_sphere(self.embed(x_full))
            h = norm_sphere(self.embed_proj(e))
        else:
            h = norm_sphere(self.embed(x_full))

        states = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            h, states[i] = block(h, states[i])

        if self.head_proj is not None:
            h_out = norm_sphere(self.head_proj(h))
        else:
            h_out = h
        return self.head(h_out)

    def get_aux_loss(self):
        loss = 0.0
        for m in self.modules():
            if isinstance(m, CausalFFTBlock) and m.use_hippocampus:
                loss = loss + m.get_phase_loss()
        return self.args.lambda_phase * loss
