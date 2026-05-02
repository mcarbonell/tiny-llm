"""
model_lateral_v197.py — Lateral Interaction & Child Neurons (V197-V198 Era)

Este módulo implementa el concepto de "Neuronas Hijas": neuronas que nacen
de la interacción simbólica de neuronas de la capa anterior.

Componentes:
1. Gumbel-Softmax Gating: Para elegir operaciones discretas (+, -, *, %) de forma diferenciable.
2. Cross-Neuron interaction: Las neuronas i y j se combinan para resolver leyes complejas.
3. Residual Integration: Se integra perfectamente con el flujo de Neurogénesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LateralInteractionLayer(nn.Module):
    """
    Capa de Interacción Lateral (V197).
    En lugar de un MLP denso, esta capa busca pares de 'neuronas padres' 
    y las combina usando una operación lógica seleccionada dinámicamente.
    """
    def __init__(self, dim, num_interactions=128, temperature=1.0):
        super().__init__()
        self.dim = dim
        self.num_interactions = num_interactions
        self.temperature = temperature # Para Gumbel-Softmax

        # Seleccionadores de 'Padres' (Proyecciones lineales para elegir qué rasgos combinar)
        self.parent_a = nn.Linear(dim, num_interactions, bias=False)
        self.parent_b = nn.Linear(dim, num_interactions, bias=False)
        
        # Selector de Operación: {SUM, DIFF, PROD, MOD_approx}
        # Usamos 4 canales para las 4 operaciones simbólicas
        self.op_selector = nn.Parameter(torch.randn(num_interactions, 4))
        
        # Proyección de salida para reintegrar al flujo principal
        self.out_proj = nn.Linear(num_interactions, dim, bias=False)
        
        # Inicialización a cero (Regla V170) para inserción residual suave
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        # x: (bsz, seq_len, dim)
        
        # 1. Obtener los valores de los padres
        a = self.parent_a(x) # (bsz, seq_len, num_interactions)
        b = self.parent_b(x)
        
        # 2. Seleccionar Operación mediante Gumbel-Softmax (Diferenciable pero tiende a discreto)
        # logits: (num_interactions, 4)
        if self.training:
            op_weights = F.gumbel_softmax(self.op_selector, tau=self.temperature, hard=True)
        else:
            op_weights = F.softmax(self.op_selector / self.temperature, dim=-1)
            # En inferencia podemos forzar hard-max
            indices = op_weights.max(dim=-1, keepdim=True)[1]
            op_weights = torch.zeros_like(op_weights).scatter_(-1, indices, 1.0)

        # 3. Calcular las 4 operaciones posibles
        # Usamos aproximaciones continuas para MOD y PROD estable
        res_sum  = a + b
        res_diff = a - b
        res_prod = a * b
        res_mod  = a - b * torch.floor(a / (b + 1e-6) + 1e-6) # Aproximación STE-like del hallazgo V195
        
        # Stack de resultados: (bsz, seq_len, num_interactions, 4)
        combined_ops = torch.stack([res_sum, res_diff, res_prod, res_mod], dim=-1)
        
        # 4. Aplicar el gating de operación (Dot product con los pesos de Gumbel)
        # op_weights: (num_interactions, 4) -> expandir para el batch
        child_neurons = (combined_ops * op_weights).sum(dim=-1)
        
        # 5. Salida proyectada
        return self.out_proj(child_neurons)

class ResidualLateralBlock(nn.Module):
    """Un bloque que envuelve la interacción lateral con normalización y residuo."""
    def __init__(self, dim, num_interactions=128):
        super().__init__()
        self.norm = nn.RMSNorm(dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(dim)
        self.lateral = LateralInteractionLayer(dim, num_interactions)

    def forward(self, x):
        # Flujo: x + Lateral(Norm(x))
        return x + self.lateral(self.norm(x))
