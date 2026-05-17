import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class SpectralArgsV10:
    dim: int = 128
    n_layers: int = 6
    vocab_size: int = 32768
    max_seq_len: int = 1024
    k_walsh: int = 64
    k_mem: int = 32      # Tamaño del Hipocampo por capa
    chunk_size: int = 256 # Tamaño de los bloques BPTT
    gamma: float = 0.9    # Retención de memoria
    lambda_phase: float = 0.01

# ══════════════════════════════════════════════════════════════════════
# nGPT & SPHERICAL UTILS
# ══════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════
# WALSH/HADAMARD UTILS
# ══════════════════════════════════════════════════════════════════════
def get_walsh_matrix_1d(dim):
    if dim == 1:
        return torch.tensor([[1.]])
    H = get_walsh_matrix_1d(dim // 2)
    return torch.cat([
        torch.cat([H, H], dim=1),
        torch.cat([H, -H], dim=1)
    ], dim=0) / math.sqrt(2)

class WalshLinear(nn.Module):
    def __init__(self, in_features, out_features, k, normalized=True):
        super().__init__()
        self.k = min(k, in_features, out_features)
        self.core = nn.Parameter(torch.randn(self.k, self.k) / math.sqrt(self.k))
        self.scale = nn.Parameter(torch.ones(1)) if normalized else None
        
        self.register_buffer('H_in', get_walsh_matrix_1d(in_features))
        self.register_buffer('H_out', get_walsh_matrix_1d(out_features))

    def forward(self, x):
        W_synthesized = self.H_out[:, :self.k] @ self.core @ self.H_in[:self.k, :]
        if self.scale is not None:
            w = F.normalize(W_synthesized, dim=-1)
            return F.linear(x, w) * self.scale
        else:
            return F.linear(x, W_synthesized)

# ══════════════════════════════════════════════════════════════════════
# STATEFUL MIXER (FOURIER HIPPOCAMPUS)
# ══════════════════════════════════════════════════════════════════════
class StatefulComplexFFTMixer(nn.Module):
    def __init__(self, T, D, k_walsh, k_mem, gamma):
        super().__init__()
        self.T = T
        self.pad_T = 1
        while self.pad_T < 2*T: self.pad_T *= 2
        self.n_freq = self.pad_T // 2 + 1
        
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        
        mask = torch.zeros(self.pad_T)
        mask[:T] = 1.0
        self.register_buffer('causal_mask', mask)
        self.out_proj = WalshLinear(D, D, k_walsh, normalized=True)
        
        # Hippocampus
        self.k_mem = min(k_mem, self.n_freq)
        self.gamma = gamma
        self.read_gate  = nn.Parameter(torch.ones(self.k_mem, 1, dtype=torch.complex64))
        self.write_gate = nn.Parameter(torch.ones(self.k_mem, 1, dtype=torch.complex64))

    def forward(self, x, memory_state=None):
        B, T, D = x.shape
        xt = x.permute(0, 2, 1) # B, D, T
        pad = torch.zeros(B, D, self.pad_T-T, device=x.device)
        xt_pad = torch.cat([xt, pad], dim=-1)
        
        X = torch.fft.rfft(xt_pad, dim=-1) # B, D, freq

        # Causal Gate
        gate_raw  = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        h_raw     = torch.fft.irfft(gate_raw, n=self.pad_T)
        h_causal  = h_raw * self.causal_mask
        gate_causal = torch.fft.rfft(h_causal, n=self.pad_T)
        
        X_gated = X * gate_causal
        X_gated_perm = X_gated.permute(0, 2, 1) # B, freq, D
        
        # READ
        if memory_state is not None:
            X_gated_perm[:, :self.k_mem, :] += memory_state * self.read_gate
            
        X_gated = X_gated_perm.permute(0, 2, 1)
        
        # IFFT
        out = torch.fft.irfft(X_gated, n=self.pad_T, dim=-1)[..., :T]
        out = out.permute(0, 2, 1)
        
        # WRITE
        current_mem = X_gated_perm[:, :self.k_mem, :].clone()
        if memory_state is None:
            new_memory_state = current_mem * self.write_gate
        else:
            new_memory_state = self.gamma * memory_state + (1 - self.gamma) * (current_mem * self.write_gate)
            
        return self.out_proj(out), new_memory_state

    def get_phase_loss(self):
        diffs = self.phase[1:] - self.phase[:-1]
        return torch.mean(torch.abs(diffs))

class NarrowFFN(nn.Module):
    def __init__(self, D, k_walsh):
        super().__init__()
        self.proj = WalshLinear(D, D, k_walsh, normalized=True)
    def forward(self, x):
        return F.gelu(self.proj(x))

class nGPTBlockStateful(nn.Module):
    def __init__(self, args: SpectralArgsV10):
        super().__init__()
        self.mixer = StatefulComplexFFTMixer(args.chunk_size, args.dim, args.k_walsh, args.k_mem, args.gamma)
        self.ffn = NarrowFFN(args.dim, args.k_walsh)
        self.alpha_m = nn.Parameter(torch.full((args.dim,), 0.05))
        self.alpha_f = nn.Parameter(torch.full((args.dim,), 0.05))

    def forward(self, x, state=None):
        m, next_state = self.mixer(x, state)
        m = norm_sphere(m)
        x = norm_sphere(x + self.alpha_m.abs().unsqueeze(0).unsqueeze(0) * m)
        
        f = norm_sphere(self.ffn(x))
        x = norm_sphere(x + self.alpha_f.abs().unsqueeze(0).unsqueeze(0) * f)
        return x, next_state

# ══════════════════════════════════════════════════════════════════════
# SPECTRAL THINKER V10
# ══════════════════════════════════════════════════════════════════════
class SpectralThinkerV10(nn.Module):
    def __init__(self, args: SpectralArgsV10):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.blocks = nn.ModuleList([nGPTBlockStateful(args) for _ in range(args.n_layers)])
        self.head = SphericalHead(args.dim, args.vocab_size, init_tau=10.0)

    def forward(self, x_full):
        """
        Divide la secuencia x_full (e.g. 1024 tokens) en bloques de chunk_size (e.g. 256)
        y pasa el estado (Hipocampo) internamente para mantener la transparencia en train.py.
        """
        B, total_len = x_full.shape
        chunk_size = self.args.chunk_size
        
        # Si la secuencia no es divisible, rellenamos o cortamos. Aquí asumimos que lo es.
        num_chunks = max(1, total_len // chunk_size)
        
        h_full = norm_sphere(self.embed(x_full))
        
        out_chunks = []
        states = [None] * len(self.blocks)
        
        for c in range(num_chunks):
            # Calculamos índices reales para no pasarnos de total_len en el último chunk
            start = c * chunk_size
            end = min((c + 1) * chunk_size, total_len)
            
            # Si el último chunk es más pequeño que chunk_size, el mixer de Fourier
            # puede fallar si está forzado a T_chunk estricto. 
            # Como train.py siempre pasa max_seq_len, asumimos end - start == chunk_size
            h_chunk = h_full[:, start:end, :]
            
            for i, block in enumerate(self.blocks):
                h_chunk, states[i] = block(h_chunk, states[i])
                
            out_chunks.append(h_chunk)
            
        h_final = torch.cat(out_chunks, dim=1)
        return self.head(h_final)

    def get_aux_loss(self):
        """Recupera la regularización de fase de todos los mixers."""
        loss_phase = 0.0
        for m in self.modules():
            if isinstance(m, StatefulComplexFFTMixer):
                loss_phase += m.get_phase_loss()
        return self.args.lambda_phase * loss_phase
