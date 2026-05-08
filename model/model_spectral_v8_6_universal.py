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

try:
    from kernels.fwht_op import fwht_native
except ImportError:
    fwht_native = None

class FastUniversalFWHT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        device = x.device
        
        # Offload a CPU para el kernel ultrarrápido si está disponible
        if fwht_native is not None:
            x_cpu = x_flat.cpu() if device.type != 'cpu' else x_flat.clone()
            res = fwht_native(x_cpu)
            if res is not None:
                return res.to(device).view(orig_shape)
                
        # Fallback a vectorizado si no hay kernel
        if device.type != 'cpu':
            return fwht_gpu_vectorized(x)
        return fwht_cpu_best(x)

    @staticmethod
    def backward(ctx, grad_output):
        orig_shape = grad_output.shape
        grad_flat = grad_output.view(-1, orig_shape[-1])
        device = grad_output.device
        
        if fwht_native is not None:
            grad_cpu = grad_flat.cpu() if device.type != 'cpu' else grad_flat.clone()
            res = fwht_native(grad_cpu)
            if res is not None:
                return res.to(device).view(orig_shape)
                
        if device.type != 'cpu':
            return fwht_gpu_vectorized(grad_output)
        return fwht_cpu_best(grad_output)

def fwht_universal(x):
    """Selecciona el motor de FWHT según el dispositivo del tensor."""
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
