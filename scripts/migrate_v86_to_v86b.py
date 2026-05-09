"""
migrate_v86_to_v86b.py — Script para inyectar pesos de v8.6 en v8.6b.
Inicializa los gates a 1.0 para preservar el comportamiento exacto.
"""

import torch
import os
import sys

# Añadir ruta base
sys.path.append(os.getcwd())

from model.model_spectral_v8_6b_gated import SpectralThinkerV8_6b, SpectralArgs

def migrate(checkpoint_path, output_path):
    print(f"Cargando checkpoint original: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extraer args y crear modelo v8.6b
    args = checkpoint['args']
    model_b = SpectralThinkerV8_6b(args)
    
    # Obtener state_dict original
    old_sd = checkpoint['model']
    new_sd = model_b.state_dict()
    
    print("Mapeando pesos...")
    
    # Lista de nombres de parámetros en el nuevo modelo que son GATES (inicializar a 1.0)
    # y nombres de parámetros que vienen del modelo viejo (copiar)
    mapped_count = 0
    gate_count = 0
    
    for name in new_sd.keys():
        if "gate" in name or "alpha" in name or "beta" in name:
            # Es un nuevo parámetro de gating, lo ponemos a 1.0
            new_sd[name] = torch.ones_like(new_sd[name])
            gate_count += 1
        elif name in old_sd:
            # El parámetro existía, lo copiamos
            new_sd[name] = old_sd[name]
            mapped_count += 1
        else:
            # Caso especial: SpectralLinear cambió a GatedSpectralLinear
            # old: layers.0.hra.q_filter.diag
            # new: layers.0.hra.q_filter.diag  (sigue igual, solo se añadió .gate)
            # Si hay alguna discrepancia de nombres por el cambio de clase, se maneja aquí
            print(f"⚠️ Aviso: {name} no encontrado en checkpoint original")

    print(f"Copiados {mapped_count} parámetros.")
    print(f"Inicializados {gate_count} nuevos gates a 1.0.")
    
    # Guardar nuevo checkpoint
    checkpoint['model'] = new_sd
    checkpoint['model_file'] = "model/model_spectral_v8_6b_gated.py"
    # Reseteamos el optimizador porque la Fase 1 requiere uno nuevo (pocos parámetros)
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    
    torch.save(checkpoint, output_path)
    print(f"✅ Migración completada: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python migrate_v86_to_v86b.py <ckpt_v86.pt> <ckpt_v86b_output.pt>")
    else:
        migrate(sys.argv[1], sys.argv[2])
