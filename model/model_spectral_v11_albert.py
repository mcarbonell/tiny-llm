import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Importamos los componentes estables y validados de V10 para mantener la compatibilidad y DRY
from model.model_spectral_v10_hippocampus import (
    norm_sphere,
    get_walsh_matrix_1d,
    SphericalHead,
    StatefulComplexFFTMixer,
    nGPTBlockStateful
)

@dataclass
class SpectralArgsV11:
    dim: int = 512
    emb_dim: int = 128     # Dimensión de embedding factorizada (E)
    n_layers: int = 6
    vocab_size: int = 32768
    max_seq_len: int = 1024
    k_walsh: int = 64
    k_mem: int = 32        # Tamaño de memoria por capa virtual
    chunk_size: int = 256  # Tamaño de los bloques BPTT
    gamma: float = 0.9     # Retención de memoria del Hipocampo
    lambda_phase: float = 0.01

class SpectralThinkerV11(nn.Module):
    def __init__(self, args: SpectralArgsV11):
        super().__init__()
        self.args = args
        
        # 1. Embeddings Factorizados (E = emb_dim, d = dim)
        self.embed = nn.Embedding(args.vocab_size, args.emb_dim)
        self.embed_proj = nn.Linear(args.emb_dim, args.dim, bias=False)
        
        # Precomputamos la matriz de Walsh global una sola vez para compartirla
        H_walsh = get_walsh_matrix_1d(args.dim)
        self.register_buffer('H_global', H_walsh)
        
        # 2. Bloque nGPT Compartido (Cross-Layer Parameter Sharing)
        # Se define una única instancia que se llamará n_layers veces
        self.block = nGPTBlockStateful(args, H_global=self.H_global)
        
        # 3. Cabezal de Salida Factorizado y Vinculado (Weight Tying)
        self.head_proj = nn.Linear(args.dim, args.emb_dim, bias=False)
        self.head = SphericalHead(args.emb_dim, args.vocab_size, init_tau=10.0)
        
        # Vinculación estricta de pesos (Embedding/Head weight sharing)
        self.head.weight = self.embed.weight

    def forward(self, x_full):
        """
        Divide la secuencia x_full en bloques de chunk_size y pasa el estado 
        de memoria de forma aislada para cada capa virtual.
        """
        B, total_len = x_full.shape
        chunk_size = self.args.chunk_size
        num_chunks = max(1, total_len // chunk_size)
        
        # Embeddings factorizados con proyecciones esféricas
        e_full = norm_sphere(self.embed(x_full))
        h_full = norm_sphere(self.embed_proj(e_full))
        
        out_chunks = []
        # Mantenemos un array de estados de memoria (Fourier Hippocampus) independientes por capa virtual
        states = [None] * self.args.n_layers
        
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, total_len)
            h_chunk = h_full[:, start:end, :]
            
            # Pasada a través del bloque compartido iterando n_layers veces
            for i in range(self.args.n_layers):
                h_chunk, states[i] = self.block(h_chunk, states[i])
                
            out_chunks.append(h_chunk)
            
        h_final = torch.cat(out_chunks, dim=1)
        
        # Cabezal factorizado con proyecciones esféricas
        e_final = norm_sphere(self.head_proj(h_final))
        return self.head(e_final)

    def get_aux_loss(self):
        """Recupera la regularización de fase de todos los mixers."""
        loss_phase = 0.0
        for m in self.modules():
            if isinstance(m, StatefulComplexFFTMixer):
                loss_phase += m.get_phase_loss()
        return self.args.lambda_phase * loss_phase
