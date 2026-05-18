import torch
import sys
import os

sys.path.append(os.getcwd())
from model.model_spectral_v10_hippocampus import SpectralThinkerV10, SpectralArgsV10

print("=== SMOKE TEST: SPECTRAL THINKER V10 CON KERNEL FUSIONADO ===")

print("Instanciando model con SpectralArgsV10...")
args = SpectralArgsV10(
    dim=128,
    n_layers=2,
    vocab_size=1000,
    max_seq_len=256,
    k_walsh=32,
    k_mem=16,
    chunk_size=64
)

model = SpectralThinkerV10(args)
print("¡Construcción del modelo exitosa!")

# Crear entradas de prueba
x = torch.randint(0, 1000, (2, 64), dtype=torch.long)
print(f"Forma de entrada (x): {x.shape}")

# Forward pass
print("Ejecutando Forward pass...")
logits = model(x)
print(f"Forma de salida (logits): {logits.shape}")

# Backward pass
print("Ejecutando Backward pass...")
loss = logits.sum()
loss.backward()
print("¡Backward pass completado exitosamente sin errores de gradiente!")
print("=============================================================")
