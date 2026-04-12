import os
import json
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

def prepare_v1():
    # 1. Rutas y Configuración
    TOKENIZER_PATH = "model/tokenizer_v1.json"
    OUTPUT_BIN = "data/train_v1.bin"
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_token = "<|endoftext|>"
    all_token_ids = []

    # 2. Procesar Archivos de Texto Plano (FineWeb, Cosmopedia, TinyStories)
    txt_files = [
        ("data/raw/fineweb_edu.txt", 1.0), # Procesamos todo lo descargado
        ("data/raw/cosmopedia.txt", 1.0),
        ("data/raw/tinystories_v2.txt", 1.0)
    ]

    print("\n--- Procesando archivos de texto plano ---")
    for file_path, ratio in txt_files:
        if not os.path.exists(file_path):
            print(f"⚠️ Saltando {file_path} (no encontrado)")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Dividimos por el tag que pusimos al descargar
            docs = content.split("<|endoftext|>")
            print(f"Tokenizando {len(docs)} documentos de {os.path.basename(file_path)}...")
            
            for doc in tqdm(docs):
                text = doc.strip()
                if text:
                    ids = tokenizer.encode(text + eos_token).ids
                    all_token_ids.extend(ids)

    # 3. Procesar Archivos Sintéticos (Lógica Niveles 0-4)
    print("\n--- Procesando datos sintéticos de Lógica ---")
    logic_files = [
        "data/raw/synthetic_logic_foundation_rich_level0_dedup.jsonl",
        "data/raw/synthetic_logic_foundation_rich_level1_dedup.jsonl",
        "data/raw/synthetic_logic_foundation_rich_level2_dedup.jsonl",
        "data/raw/synthetic_logic_foundation_rich_level3_dedup.jsonl",
        "data/raw/synthetic_logic_foundation_rich_level4_dedup.jsonl",
    ]

    for file_path in logic_files:
        if not os.path.exists(file_path):
            print(f"⚠️ Saltando {file_path}")
            continue
            
        print(f"Procesando {os.path.basename(file_path)}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                data = json.loads(line)
                level = data.get("level", 0)
                raw_text = data.get("text", "")
                
                # Formatear como conversación
                # Separamos Question y Answer (asumiendo formato estándar del generador)
                parts = raw_text.split("Answer:", 1)
                if len(parts) == 2:
                    q_part = parts[0].replace("Question:", "").strip()
                    a_part = parts[1].strip()
                    formatted_text = f"User: <level_{level}> {q_part}\nAssistant: {a_part}{eos_token}"
                else:
                    formatted_text = f"User: <level_{level}>\n{raw_text}{eos_token}"
                
                ids = tokenizer.encode(formatted_text).ids
                all_token_ids.extend(ids)

    # 4. Procesar Archivos de Planificación
    print("\n--- Procesando datos sintéticos de Planificación ---")
    plan_file = "data/raw/synthetic_planning_full_v1.jsonl"
    if os.path.exists(plan_file):
        with open(plan_file, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                data = json.loads(line)
                raw_text = data.get("text", "")
                # Planificación suele ser nivel alto (ej: level 4 o 5)
                formatted_text = f"User: <level_4> Planifica lo siguiente: {raw_text}{eos_token}"
                ids = tokenizer.encode(formatted_text).ids
                all_token_ids.extend(ids)

    # 5. Guardar Binario
    print(f"\n--- Finalizando preparación ---")
    token_arr = np.array(all_token_ids, dtype=np.uint16)
    token_arr.tofile(OUTPUT_BIN)
    
    print(f"✅ Archivo binario guardado en: {OUTPUT_BIN}")
    print(f"Total de tokens: {len(token_arr):,}")
    print(f"Tamaño del archivo: {os.path.getsize(OUTPUT_BIN) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    prepare_v1()
