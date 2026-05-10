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
    Minimizamos la creación de tensores intermedios usando cat en lugar de stack.
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    b = x.numel() // n
    x = x.view(b, n)
    
    h = 1
    while h < n:
        x = x.view(b, n // (2 * h), 2, h)
        a = x[:, :, 0]
        b_ = x[:, :, 1]
        # cat es más amigable para la gestión de memoria de DirectML que stack
        x = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
        
    return x.view(orig_shape) / (n ** 0.5)

try:
    from kernels.fwht_op import fwht_native
except ImportError:
    fwht_native = None

class FastUniversalFWHT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Encapsulamos el FWHT para que PyTorch lo vea como una op atómica.
        # No guardamos nada en ctx porque el FWHT es su propio gradiente.
        if x.device.type != 'cpu':
            return fwht_gpu_vectorized(x)
        return fwht_cpu_best(x)

    @staticmethod
    def backward(ctx, grad_output):
        # El gradiente de la FWHT(x) es simplemente FWHT(grad_output) 
        # (ya que es lineal y ortogonal).
        # RETORNO: Solo un valor (el gradiente para x).
        grad_x = fwht_gpu_vectorized(grad_output) if grad_output.device.type != 'cpu' else fwht_cpu_best(grad_output)
        return grad_x

def fwht_universal(x):
    """Motor de FWHT atómico para máxima eficiencia de RAM y GPU."""
    # Caso especial CPU Nativo (inyectado vía ctypes)
    if x.device.type == 'cpu' and fwht_native is not None:
        # El kernel nativo es in-place, clonamos para mantener el grafo
        x_copy = x.clone()
        res = fwht_native(x_copy)
        if res is not None: return res
        
    return FastUniversalFWHT.apply(x)

class SpectralThinkerV8_6(SpectralThinkerV8_5):
    """
    Extiende la V8.5 con soporte universal de aceleración.
    """
    def forward(self, tokens, targets=None, holograms=None, pos=0, use_cache=False):
        # Asegurar que los tokens están en el dispositivo correcto
        device = self.codes.device
        if tokens.device != device:
            tokens = tokens.to(device)
            
        # 1. Entrada: Espacial -> Espectral
        try:
            z = F.embedding(tokens, self.codes)
        except Exception as e:
            print(f"DEBUG - tokens device: {tokens.device}, type: {type(tokens)}, dtype: {tokens.dtype}")
            print(f"DEBUG - codes device: {self.codes.device}, type: {type(self.codes)}, dtype: {self.codes.dtype}")
            raise e
        h_spatial = torch.matmul(z, self.basis)
        h_spec = fwht_universal(h_spatial)
        
        # 2. Loop de Capas: 100% Espectral
        new_holograms = []
        for i, layer in enumerate(self.layers):
            prev_h = holograms[i] if holograms is not None else None
            # Soporte para gradient checkpointing
            if getattr(layer, 'use_checkpoint', False) and self.training:
                # torch.utils.checkpoint no soporta kwargs en todas las versiones, pasamos 'pos' como posicional
                h_spec, new_h = torch.utils.checkpoint.checkpoint(layer, h_spec, prev_h, pos, use_reentrant=False)
            else:
                h_spec, new_h = layer(h_spec, prev_h, pos)
            new_holograms.append(new_h)
            
        # 3. Salida: Espectral -> Espacial
        h_final_spec = self.norm_final(h_spec)
        h_final_spatial = fwht_universal(h_final_spec)
        
        # Proyección de Logits
        latent_h = torch.matmul(h_final_spatial, self.basis.t())
        logits = torch.matmul(latent_h, self.codes.t())
        
        if targets is not None:
            if targets.device != device:
                targets = targets.to(device)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
            
        return (logits, new_holograms) if use_cache else logits
