
import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import time

# Configuración
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"
TOKENIZER_PATH = "model/tokenizer_v2_32k.json"
OUTPUT_PATH = "data/fineweb_edu_10b.bin"
CHUNK_SIZE = 5_000_000  # Escribir a disco cada 5M de tokens

def main():
    print(f"🚀 Iniciando preparación de {DATASET_NAME} ({SUBSET})...")
    
    # 1. Cargar Tokenizador
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ Error: No se encuentra el tokenizador en {TOKENIZER_PATH}")
        return
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<|endoftext|>") or 0
        
    print(f"✅ Tokenizador cargado. EOS ID: {eos_id}")

    # 2. Cargar Dataset en modo Streaming
    print("📥 Cargando dataset desde HuggingFace (modo streaming)...")
    dataset = load_dataset(DATASET_NAME, name=SUBSET, split="train", streaming=True)

    # 3. Procesamiento y Guardado
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    total_tokens = 0
    buffer = []
    start_time = time.time()
    
    print(f"✍️ Escribiendo en: {OUTPUT_PATH}")
    
    try:
        with open(OUTPUT_PATH, "wb") as f:
            # Usamos un iterador sobre el dataset
            for i, example in enumerate(dataset):
                text = example["text"]
                
                # Tokenizar
                tokens = tokenizer.encode(text).ids
                tokens.append(eos_id)
                buffer.extend(tokens)
                
                # Guardar en disco si superamos el tamaño del chunk
                if len(buffer) >= CHUNK_SIZE:
                    arr = np.array(buffer, dtype=np.uint16)
                    f.write(arr.tobytes())
                    total_tokens += len(buffer)
                    
                    # Estadísticas
                    elapsed = time.time() - start_time
                    tps = total_tokens / elapsed if elapsed > 0 else 0
                    print(f"   [{i:9d} docs] Procesados {total_tokens/1e6:6.1f}M tokens... ({tps/1e3:5.1f}k tokens/s)")
                    
                    buffer = []
                    
            # Guardar remanente
            if buffer:
                arr = np.array(buffer, dtype=np.uint16)
                f.write(arr.tobytes())
                total_tokens += len(buffer)

    except KeyboardInterrupt:
        print("\n🛑 Proceso interrumpido por el usuario.")
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")

    end_time = time.time()
    print("\n" + "="*40)
    print("✨ PROCESO COMPLETADO")
    print(f"   Tokens totales: {total_tokens/1e9:.2f} Billones")
    print(f"   Archivo final:  {OUTPUT_PATH}")
    print(f"   Tamaño aprox:   {os.path.getsize(OUTPUT_PATH)/1e9:.2f} GB")
    print(f"   Tiempo total:   {(end_time - start_time)/3600:.2f} horas")
    print("="*40)

if __name__ == "__main__":
    main()
