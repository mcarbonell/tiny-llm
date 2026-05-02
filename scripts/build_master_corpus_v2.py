"""
scripts/build_master_corpus_v2.py — The Great Data Merger

Mezcla todos los datasets descargados en un solo archivo TXT para entrenar
el Tokenizador Maestro de 32k.
"""

import os
import json
import random

def main():
    raw_dir = "data/raw"
    output_path = "data/raw/master_corpus_v2.txt"
    
    # Lista de archivos y su "peso" (cuánto texto extraer de cada uno)
    # Algunos son TXT puros, otros JSONL (donde extraemos el campo 'text')
    files = [
        ("wiki.txt", "txt", 1.0),            # 100% de Wikipedia (es pequeño)
        ("fineweb_edu.txt", "txt", 0.5),      # 50% de FineWeb (es masivo)
        ("cosmopedia.txt", "txt", 0.5),       # 50% de Cosmopedia
        ("tinystories.txt", "txt", 0.3),      # 30% de historias
        ("chat_alpaca_cleaned.jsonl", "jsonl", 1.0),
        ("chat_dolly_15k.jsonl", "jsonl", 1.0),
        ("chat_oasst_guanaco.jsonl", "jsonl", 1.0),
        ("code_alpaca.jsonl", "jsonl", 1.0),
        ("code_python_18k.jsonl", "jsonl", 1.0),
        ("synthetic_planning_full_v1.jsonl", "jsonl", 1.0),
    ]

    print(f"🚀 Creando Master Corpus V2 en {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as out:
        for filename, ftype, ratio in files:
            path = os.path.join(raw_dir, filename)
            if not os.path.exists(path):
                print(f"⚠️ Saltando {filename} (no encontrado)")
                continue
                
            print(f"📖 Procesando {filename} (ratio={ratio})...")
            count = 0
            
            if ftype == "txt":
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if random.random() < ratio:
                            out.write(line)
                            count += 1
            else:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get("text", "")
                            if text and random.random() < ratio:
                                out.write(text + "\n")
                                count += 1
                        except:
                            continue
            
            print(f"   ✅ Extraídas {count} líneas.")

    print(f"\n✨ Master Corpus V2 completado: {os.path.getsize(output_path) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
