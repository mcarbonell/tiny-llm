import os
import json
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

# Rutas
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")
INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_logic_rich.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_logic_rich.bin")

def main():
    if not os.path.exists(TOKENIZER_PATH):
        print("❌ Error: No se encuentra el tokenizador.")
        return
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: No se encuentra el archivo de entrada {INPUT_PATH}")
        return

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<eos>") or 0
    
    all_tokens = []
    print(f"📂 Cargando y tokenizando {INPUT_PATH}...")
    
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        try:
            data = json.loads(line)
            # Añadimos el prefijo de sistema para el entrenamiento
            text = f"[SYSTEM] Reasoning Engine [/SYSTEM] {data['text']}"
            tokens = tokenizer.encode(text).ids
            tokens.append(eos_id)
            all_tokens.extend(tokens)
        except Exception as e:
            print(f"⚠️ Error procesando línea: {e}")
            continue
            
    # Convertir a binario
    print(f"💾 Guardando {len(all_tokens)} tokens en {OUTPUT_PATH}...")
    np.array(all_tokens, dtype=np.uint16).tofile(OUTPUT_PATH)
    print(f"✅ ¡Hecho! El archivo binario está listo para el entrenamiento.")

if __name__ == "__main__":
    main()
