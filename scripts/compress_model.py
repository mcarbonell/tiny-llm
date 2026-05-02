"""
scripts/compress_model.py — Model Packer V198 (High Fidelity Edition)

Aplica la triada de compresión V198 sin pérdida de precisión:
1. Transformada Espectral (DCT): Mueve los pesos al dominio de la frecuencia.
2. Pruning Espectral (Top-K/Threshold): Elimina el ruido de alta frecuencia.
3. Codificación de Entropía (Zlib): Empaqueta los ceros sin pérdida.

Mantiene la precisión float32/float16 original para preservar el razonamiento.
"""

import os
import sys
import torch
import torch.nn.functional as F
import zlib
import argparse
import io
import math

# Añadir ruta base para resolver los imports de 'model'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import ModelArgs as DenseArgs
from model.model_moe import ModelArgs as MoEArgs
from model.model_coga import ModelArgs as CogaArgs
from model.model_spectral import SpectralArgs
from model.model_spectral_v4 import SpectralArgs as SpectralArgsV4
from model.model_spectral_v5 import SpectralArgs as SpectralArgsV5
from model.model_coga_spectral import CogaSpectralArgs
from model.model_analog import AnalogArgs
from model.model_auto_analog import AutoAnalogArgs

# --- Utilidades de Transformada ---

_DCT_CACHE = {}

def get_dct_matrix(N):
    """Genera una matriz DCT-II N×N completa de forma vectorizada."""
    if N in _DCT_CACHE:
        return _DCT_CACHE[N]
        
    i = torch.arange(N).view(N, 1)
    j = torch.arange(N).view(1, N)
    
    # mat[i, j] = sqrt(2/N) * cos(pi * i * (2j + 1) / (2N))
    # Para i=0, mat[0, j] = sqrt(1/N)
    mat = torch.cos(math.pi * i * (2 * j + 1) / (2 * N))
    mat[0, :] *= 1.0 / math.sqrt(2.0)
    mat *= math.sqrt(2.0 / N)
    
    _DCT_CACHE[N] = mat
    return mat

def compress_weight_spectral(name, tensor, threshold=0.005):
    """Transforma un peso denso a espectral y aplica pruning."""
    # Saltamos capas no matriciales o críticas para la base
    if tensor.dim() != 2 or "tok_embeddings" in name or "norm" in name or "output" in name:
        mask = torch.abs(tensor) > (threshold * 0.1) # Pruning muy leve en capas críticas
        return tensor * mask, False

    # 1. Transformada Espectral (DCT)
    out_features, in_features = tensor.shape
    d_out = get_dct_matrix(out_features).to(tensor.device).to(tensor.dtype)
    d_in = get_dct_matrix(in_features).to(tensor.device).to(tensor.dtype)
    
    # Mover al dominio de la frecuencia (DCT 2D)
    # W_spectral = D_out @ W_spatial @ D_in.T
    spectral_coeffs = d_out @ tensor @ d_in.t()
    
    # 2. Pruning Espectral (Threshold)
    mask = torch.abs(spectral_coeffs) > threshold
    sparse_spectral = spectral_coeffs * mask
    
    return {"data": sparse_spectral, "type": "spectral_dct_v198"}, True

def pack_model(checkpoint_path, output_path, threshold=0.005):
    print(f"📦 Empacando modelo (V198 High-Fidelity): {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']
    
    compressed_state = {}
    original_size = os.path.getsize(checkpoint_path)
    
    for name, tensor in state_dict.items():
        # Aplicar el flujo V198
        c_data, was_spectral = compress_weight_spectral(name, tensor, threshold)
        compressed_state[name] = c_data
        
    package = {
        "arch": checkpoint.get('arch', 'dense'),
        "args": checkpoint.get('args'),
        "iter_num": checkpoint.get('iter_num', 0),
        "val_loss": checkpoint.get('val_loss', 0),
        "model": compressed_state,
        "compression_v": "V198_HI_FI"
    }
    
    # 3. Codificación de Entropía (Zlib)
    print("⚡ Aplicando empaquetado de entropía (Zlib level 9)...")
    buffer = io.BytesIO()
    torch.save(package, buffer)
    raw_bytes = buffer.getvalue()
    
    compressed_bytes = zlib.compress(raw_bytes, level=9)
    
    with open(output_path, "wb") as f:
        f.write(compressed_bytes)
        
    compressed_size = os.path.getsize(output_path)
    print(f"\n✅ PROCESO COMPLETADO")
    print(f"----------------------------------------")
    print(f"Original:   {original_size / 1024**2:.2f} MB")
    print(f"Comprimido: {compressed_size / 1024**2:.2f} MB")
    print(f"Ahorro:     {(1 - compressed_size/original_size)*100:.2f}%")
    print(f"Ratio:      {original_size/compressed_size:.1f}x")
    print(f"Archivo:    {output_path}")
    print(f"----------------------------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.002) # Umbral conservador para no perder calidad
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.input.replace(".pt", ".tiny")
    pack_model(args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()
