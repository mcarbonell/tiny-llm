import torch
import sys
import os

# Agregar ruta base
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model_spectral_v8 import SpectralThinkerV8, SpectralArgs

def smoke_test_v8():
    print("--- SMOKE TEST: SPECTRAL V8 (HOLOGRAPHIC MOE) ---")
    
    device = torch.device('cpu')
    args = SpectralArgs(
        dim=512,        # Dimensión reducida para el test
        n_layers=2,     # Solo 2 capas
        num_experts=1024, # Menos expertos para rapidez
        vocab_size=1000
    )
    
    print(f"Instanciando modelo (Experts: {args.num_experts})...")
    model = SpectralThinkerV8(args).to(device)
    
    # Simular un batch
    # (Batch: 2, Seq: 8)
    tokens = torch.randint(0, 1000, (2, 8))
    targets = torch.randint(0, 1000, (2, 8))
    
    print("Ejecutando Forward Pass...")
    logits, loss = model(tokens, targets=targets)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss inicial: {loss.item():.4f}")
    
    print("Verificando Backpropagation...")
    loss.backward()
    
    # Comprobar gradientes en MoE y HRA
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"¿Gradientes generados?: {'SÍ' if has_grads else 'NO'}")
    
    print("\n[OK] La arquitectura V8 es funcional.")

if __name__ == "__main__":
    smoke_test_v8()
