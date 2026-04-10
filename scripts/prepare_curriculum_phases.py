import os
import numpy as np
import random

# Configuración de archivos base (deben existir previamente)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BASE_STORIES = os.path.join(DATA_DIR, "train.bin")
BASE_WIKI    = os.path.join(DATA_DIR, "wiki.bin")
BASE_LOGIC   = os.path.join(DATA_DIR, "synthetic_logic.bin")

def load_data(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} no encontrado.")
        return None
    return np.fromfile(path, dtype=np.uint16)

def create_phase(name, components):
    """
    components: lista de tuplas (array, peso)
    """
    print(f"--- Creando {name} ---")
    total_tokens = 5_000_000 # Objetivo: 5M de tokens por fase para el experimento inicial
    
    mixed_data = []
    for data, weight in components:
        if data is None: continue
        n_tokens = int(total_tokens * weight)
        # Tomar una muestra aleatoria o secuencial
        if len(data) > n_tokens:
            start = random.randint(0, len(data) - n_tokens)
            sub = data[start:start+n_tokens]
        else:
            sub = data # Si es menor, lo tomamos todo
        mixed_data.append(sub)
    
    final_arr = np.concatenate(mixed_data)
    # Mezclamos un poco (opcional, pero ayuda)
    output_file = os.path.join(DATA_DIR, name)
    final_arr.tofile(output_file)
    print(f"✅ Guardado en {output_file} ({len(final_arr)} tokens)")

def main():
    stories = load_data(BASE_STORIES)
    wiki    = load_data(BASE_WIKI)
    logic   = load_data(BASE_LOGIC)

    # Fase 1: 100% Historias (Gramática)
    create_phase("phase1_grammar.bin", [(stories, 1.0)])

    # Fase 2: 70% Historias + 30% Wiki (Conocimiento)
    create_phase("phase2_knowledge.bin", [(stories, 0.7), (wiki, 0.3)])

    # Fase 3: 40% Wiki + 60% Lógica (Razonamiento)
    create_phase("phase3_reasoning.bin", [(wiki, 0.4), (logic, 0.6)])

if __name__ == "__main__":
    main()
