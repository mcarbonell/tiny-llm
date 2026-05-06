"""
model_spectral_v8_6_universal.py — The "Universal Core" Architecture
Optimized for CPU (Native), AMD (DirectML), and NVIDIA (CUDA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from model.model_spectral_v8_5_native import SpectralThinkerV8_5, fwht_best as fwht_cpu_best, SpectralArgs

# Detección de aceleración
HAS_DML = False
try:
    import torch_directml
    HAS_DML = True
except ImportError:
    pass

def get_best_device():
    """Detecta el mejor hardware disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if HAS_DML:
        return torch_directml.device()
    return torch.device("cpu")

def fwht_gpu_vectorized(x):
    """
    Versión de FWHT optimizada para GPU (DirectML/CUDA).
    Usa operaciones tensoriales masivas en lugar de bucles C++.
    """
    # x: (..., N)
    orig_shape = x.shape
    n = orig_shape[-1]
    b = x.numel() // n
    x = x.view(b, n)
    
    h = 1
    while h < n:
        # Reestructuramos para operar en paralelo sobre los pares de mariposa
        x = x.view(b, n // (2 * h), 2, h)
        # En GPU, stack + sum/sub es muy eficiente
        a = x[:, :, 0]
        b_ = x[:, :, 1]
        x = torch.stack([a + b_, a - b_], dim=2)
        h *= 2
        
    return x.view(orig_shape) / (n ** 0.5)

def fwht_universal(x):
    """Selecciona el motor de FWHT según el dispositivo del tensor."""
    if x.is_cuda or (HAS_DML and "privateuseone" in str(x.device)):
        return fwht_gpu_vectorized(x)
    return fwht_cpu_best(x)

class SpectralThinkerV8_6(SpectralThinkerV8_5):
    """
    Extiende la V8.5 con soporte universal de aceleración.
    """
    def __init__(self, args: SpectralArgs):
        super().__init__(args)
        self._current_device = torch.device("cpu")

    def to_device(self, device=None):
        if device is None:
            device = get_best_device()
        self._current_device = device
        self.to(device)
        print(f"Modelo V8.6 movido a: {device}")
        return self

    def forward(self, tokens, targets=None, holograms=None, pos=0, use_cache=False):
        # Asegurar que los tokens están en el dispositivo correcto
        if tokens.device != self._current_device:
            tokens = tokens.to(self._current_device)
            
        # 1. Entrada: Espacial -> Espectral
        z = F.embedding(tokens, self.codes)
        h_spatial = torch.matmul(z, self.basis)
        h_spec = fwht_universal(h_spatial)
        
        # 2. Loop de Capas: 100% Espectral
        new_holograms = []
        for i, layer in enumerate(self.layers):
            prev_h = holograms[i] if holograms is not None else None
            # Las capas heredan el dispositivo del modelo
            h_spec, new_h = layer(h_spec, prev_h, pos)
            new_holograms.append(new_h)
            
        # 3. Salida: Espectral -> Espacial
        h_final_spec = self.norm_final(h_spec)
        h_final_spatial = fwht_universal(h_final_spec)
        
        # Proyección de Logits
        latent_h = torch.matmul(h_final_spatial, self.basis.t())
        logits = torch.matmul(latent_h, self.codes.t())
        
        if targets is not None:
            if targets.device != self._current_device:
                targets = targets.to(self._current_device)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
            
        return (logits, new_holograms) if use_cache else logits
