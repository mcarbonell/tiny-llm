import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from model.model_spectral_v8_4_optimized import SpectralThinkerV8_4, fwht_iterative, SpectralArgs

# Importamos el cargador del kernel nativo
try:
    from kernels.fwht_op import fwht_native
except ImportError:
    fwht_native = None

def fwht_best(x):
    """
    Usa el kernel nativo si está disponible, si no, usa el iterativo de v8.4.
    """
    # El kernel nativo espera (Batch, Dim)
    orig_shape = x.shape
    if len(orig_shape) > 2:
        x = x.reshape(-1, orig_shape[-1])
        
    res = None
    if fwht_native is not None:
        res = fwht_native(x)
        
    if res is None:
        res = fwht_iterative(x)
        
    return res.view(orig_shape)

class SpectralThinkerV8_5(SpectralThinkerV8_4):
    """
    Hereda la arquitectura de Residencia Espectral de la v8.4
    pero inyecta el motor nativo de FWHT vía ctypes.
    """
    def forward(self, tokens, targets=None, holograms=None, pos=0, use_cache=False):
        import torch.nn.functional as F
        
        # 1. Entrada: Espacial -> Espectral
        z = F.embedding(tokens, self.codes)
        h_spatial = torch.matmul(z, self.basis)
        h_spec = fwht_best(h_spatial)
        
        # 2. Loop de Capas: 100% Espectral
        new_holograms = []
        for i, layer in enumerate(self.layers):
            prev_h = holograms[i] if holograms is not None else None
            h_spec, new_h = layer(h_spec, prev_h, pos)
            new_holograms.append(new_h)
            
        # 3. Salida: Espectral -> Espacial
        h_final_spec = self.norm_final(h_spec)
        h_final_spatial = fwht_best(h_final_spec)
        
        # Proyección de Logits
        latent_h = torch.matmul(h_final_spatial, self.basis.t())
        logits = torch.matmul(latent_h, self.codes.t())
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
            
        return (logits, new_holograms) if use_cache else logits
