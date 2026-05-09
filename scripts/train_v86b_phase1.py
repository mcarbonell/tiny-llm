"""
train_v86b_phase1.py — Script de entrenamiento para la Fase 1 del Blueprint.
Solo entrena los GATES con un LR alto (0.01).
"""

import torch
import torch.nn as nn
import os
import sys
import time
import datetime

# Añadir ruta base
sys.path.append(os.getcwd())

from model.model_spectral_v8_6b_gated import SpectralThinkerV8_6b
from scripts.config import get_config, load_dataset

def t_print(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

def train_phase1(config_path, checkpoint_path):
    config = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Soporte para DirectML
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        pass

    t_print(f"DEVICE: {device}")
    
    # 1. Cargar el checkpoint migrado (v8.6b)
    t_print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = SpectralThinkerV8_6b(checkpoint['args'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    # 2. CONGELACIÓN (Regla del Blueprint Fase 1)
    t_print("Aplicando congelación Fase 1: Solo Gates...")
    trainable_params = []
    frozen_count = 0
    
    for name, param in model.named_parameters():
        if "gate" in name or "alpha" in name or "beta" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            frozen_count += 1
            
    t_print(f"Parámetros congelados: {frozen_count}")
    t_print(f"Parámetros entrenables (Gates): {len(trainable_params)}")

    # 3. OPTIMIZADOR (LR alto según el Blueprint)
    # Usamos un LR agresivo de 0.01 (ajustable)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-2, weight_decay=0.0)
    
    # 4. Dataset
    train_data, val_data = load_dataset(config['dataset'])
    
    # 5. Bucle de entrenamiento simplificado
    model.train()
    best_val_loss = float('inf')
    
    t_print("Iniciando Fase 1 (Structural Warmup)...")
    
    for i in range(1000): # Probaremos con 1000 iters iniciales
        # Obtener batch (esto asume que tienes una función get_batch en config o similar)
        # Por simplicidad en este prototipo, simulamos el flujo de train.py
        
        # ... (Aquí iría el bucle de entrenamiento estándar de train.py pero filtrado) ...
        # Para no duplicar 600 líneas de código, el usuario usará train.py 
        # con un flag de congelación si lo prefiere, o este script ligero.
        pass

    t_print("Script de entrenamiento Fase 1 listo para ser integrado en train.py")

if __name__ == "__main__":
    print("Este script es un PROTOTIPO. Se recomienda integrar la lógica de congelación en scripts/train.py")
