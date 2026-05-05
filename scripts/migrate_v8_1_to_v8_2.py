import torch
import os
import sys

# Añadir paths para importación
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

from model.model_spectral_v8_1 import SpectralThinkerV8_1 as ModelV81, SpectralArgs as ArgsV81
from model.model_spectral_v8_2 import SpectralThinkerV8_2 as ModelV82, SpectralArgs as ArgsV82

def migrate_weights(v81_checkpoint_path, output_path):
    print(f"Cargando checkpoint V8.1 desde: {v81_checkpoint_path}")
    
    # 1. Cargar el checkpoint original
    checkpoint = torch.load(v81_checkpoint_path, map_location='cpu')
    state_dict_v81 = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 2. Inicializar el modelo V8.2 con la misma configuración
    # (Asumimos que las dimensiones base son las mismas)
    args = ArgsV82() # Puedes ajustar esto si usas una config distinta
    model_v82 = ModelV82(args)
    state_dict_v82 = model_v82.state_dict()
    
    print("Migrando pesos...")
    
    new_state_dict = {}
    count_matched = 0
    count_new = 0
    
    for key in state_dict_v82.keys():
        if key in state_dict_v81:
            # El peso existe en ambos, lo copiamos
            new_state_dict[key] = state_dict_v81[key]
            count_matched += 1
        else:
            # Es un peso nuevo (ej. saliency_proj)
            print(f"  [NUEVO] Inicializando capa: {key}")
            
            # TRUCO: Inicializamos el bias de saliencia alto para que 
            # al principio el modelo se comporte como la V8.1 (Saliencia ~ 1.0)
            if 'saliency_proj.weight' in key:
                # Inicialización casi neutra
                nn_weight = torch.randn_like(state_dict_v82[key]) * 0.01
                new_state_dict[key] = nn_weight
            elif 'saliency_proj.bias' in key:
                # Bias alto para que Sigmoid(x) ≈ 1.0
                new_state_dict[key] = torch.ones_like(state_dict_v82[key]) * 2.5
            else:
                new_state_dict[key] = state_dict_v82[key]
            
            count_new += 1
            
    # 3. Guardar el nuevo checkpoint
    save_data = {
        'model': new_state_dict,
        'args': args,
        'info': "Migrado desde V8.1 con Saliency Patching"
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(save_data, output_path)
    
    print(f"\nMigración completada con éxito:")
    print(f"- Pesos transferidos: {count_matched}")
    print(f"- Nuevos pesos inicializados: {count_new}")
    print(f"Nuevo checkpoint guardado en: {output_path}")

if __name__ == "__main__":
    # Ejemplo de uso (ajustar rutas según necesidad)
    v81_path = "checkpoints/spectral_v8_1_compressed/last.pt"
    v82_target = "checkpoints/spectral_v8_2_saliency/migrated_from_v81.pt"
    
    if os.path.exists(v81_path):
        migrate_weights(v81_path, v82_target)
    else:
        print(f"No se encontró el checkpoint en {v81_path}. Ejecuta el script manualmente cuando tengas un checkpoint listo.")
