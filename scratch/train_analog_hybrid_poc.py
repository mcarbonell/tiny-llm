"""
scratch/train_analog_hybrid_poc.py — Prueba de Concepto (POC) del Razonamiento Híbrido

Este script genera un dataset sintético de lógica (A + B % C) y entrena el modelo
TinyThinkerAnalogHybrid utilizando el doble optimizador Adam + DGE.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import json

# Añadir ruta para importar el modelo (basada en la ubicación del script)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_analog_hybrid import TinyThinkerAnalogHybrid, AnalogArgs

# --- IMPORTAR DGE OPTIMIZER ---
DGE_PATH = r"C:\Users\mrcm_\Local\proj\algorithms\dge-optimizer"
if DGE_PATH not in sys.path:
    sys.path.append(DGE_PATH)

try:
    from dge.torch_optimizer import TorchDGEOptimizer
except ImportError:
    print(f"Error: No se pudo importar TorchDGEOptimizer. Verifica la ruta: {DGE_PATH}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- GENERACIÓN DE DATASET SINTÉTICO (LÓGICA DE MÓDULO) ---
def generate_logic_data(n_samples=2000, seq_len=8, vocab_size=20, mod=7):
    """
    Tarea: Dado un prefijo de números, predecir el acumulado MOD 7.
    Ej: [2, 3, 1] -> Target [2, 5, 6] (todos mod 7)
    """
    x = torch.randint(1, vocab_size, (n_samples, seq_len))
    y = torch.zeros_like(x)
    
    for i in range(n_samples):
        acc = 0
        for j in range(seq_len):
            acc = (acc + x[i, j]) % mod
            y[i, j] = acc
            
    return x.to(device), y.to(device)

def run_poc():
    print("--- INICIANDO POC: RAZONAMIENTO HÍBRIDO (ADAM + DGE) ---")
    
    args = AnalogArgs(
        dim=64, 
        n_layers=2, 
        n_heads=4, 
        vocab_size=20,    # Vocabulario pequeño para facilitar el aprendizaje de la identidad
        max_seq_len=16
    )
    
    model = TinyThinkerAnalogHybrid(args).to(device)
    
    # 1. SEPARAR PARÁMETROS
    dge_params = []
    adam_params = []
    
    for name, param in model.named_parameters():
        if "symbolic.w_ops" in name: # SOLO w_ops para DGE
            dge_params.append(param)
        else:
            adam_params.append(param)
            
    flat_dge = torch.nn.utils.parameters_to_vector(dge_params).detach().clone()
    dge_dim = flat_dge.numel()
    
    # 2. CONFIGURAR OPTIMIZADORES
    optimizer_adam = optim.Adam(adam_params, lr=0.002) 
    
    dge = TorchDGEOptimizer(
        dim=dge_dim,
        k_blocks=8,       # Con tan pocos parámetros, 8 bloques sobran
        lr=0.2,           # DGE muy agresivo para el ruteo
        delta=0.5,
        total_steps=2000,
        device=device
    )
    
    # 3. DATOS
    x_train, y_train = generate_logic_data(n_samples=2000)
    x_test, y_test = generate_logic_data(n_samples=500)
    
    batch_size = 64      # Batch más grande para gradientes más estables
    epochs = 2000        # Más tiempo para que DGE refine los pesos
    t0 = time.time()
    
    print(f"   Modelo: {args.n_layers} capas, {args.dim} dim")
    print(f"   Parámetros Adam: {sum(p.numel() for p in adam_params)}")
    print(f"   Parámetros DGE (Symbolic): {dge_dim}")
    print("-" * 50)

    for epoch in range(epochs):
        # Seleccionar batch aleatorio
        idx = torch.randperm(x_train.size(0))[:batch_size]
        xb, yb = x_train[idx], y_train[idx]
        
        # --- PASO DGE ---
        def f_dge(P_batch):
            P = P_batch.shape[0]
            losses = torch.empty(P, device=device)
            with torch.no_grad():
                for i in range(P):
                    torch.nn.utils.vector_to_parameters(P_batch[i], dge_params)
                    _, loss = model(xb, yb)
                    losses[i] = loss
            print(".", end="", flush=True) # Indicador de progreso del DGE
            return losses

        flat_dge, _ = dge.step(f_dge, flat_dge)
        torch.nn.utils.vector_to_parameters(flat_dge, dge_params)
        
        # --- PASO ADAM ---
        model.train()
        optimizer_adam.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer_adam.step()
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                _, test_loss = model(x_test, y_test)
                # Calcular precisión simple
                logits = model(x_test[:100])
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == y_test[:100]).float().mean().item()
                
            print(f"  Paso {epoch} | Loss: {test_loss.item():.4f} | Acc: {acc*100:.2f}%")

    # --- MÉTRICAS FINALES ---
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        preds = torch.argmax(logits, dim=-1)
        final_acc = (preds == y_test).float().mean().item()
    
    print("\n" + "="*50)
    print(f"POC FINALIZADO: TINY-THINKER ANALOG HYBRID")
    print(f"="*50)
    print(f"Precisión Final (Logic): {final_acc*100:.2f}%")
    print(f"Tiempo Total: {dt:.1f}s")
    print(f"Parámetros DGE: {dge_dim}")
    print("="*50)

if __name__ == "__main__":
    run_poc()
