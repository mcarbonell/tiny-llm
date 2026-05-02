"""
scripts/download_chat_data.py — Data Harvester for the Chat Era

Descarga datasets de alta calidad orientados a CHAT e INSTRUCCIONES:
1. Databricks Dolly 15k (Instrucciones cortas y claras)
2. OpenAssistant (Conversaciones complejas)
3. Alpaca Cleaned (Generalización de tareas)

Los guarda en data/raw/ para su posterior mezcla.
"""

import os
from datasets import load_dataset
import json

def save_as_jsonl(dataset, output_path):
    print(f"💾 Guardando en {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            # Normalizamos el formato a {"text": ...} para compatibilidad con nuestros scripts
            # Cada dataset tiene columnas diferentes, aquí las unificamos
            text = ""
            if "instruction" in entry and "output" in entry:
                # Formato Alpaca/Dolly
                input_str = entry.get("input", "")
                if input_str:
                    text = f"User: {entry['instruction']}\nContext: {input_str}\nAssistant: {entry['output']}"
                else:
                    text = f"User: {entry['instruction']}\nAssistant: {entry['output']}"
            elif "text" in entry:
                text = entry["text"]
            
            if text:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

def main():
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    print("🚀 Iniciando descarga de datasets para el 'Proyecto Chat'...")

    # 1. Dolly 15k - Muy buena calidad, conocimiento general
    print("\n--- Descargando Databricks Dolly 15k ---")
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        save_as_jsonl(dolly, os.path.join(raw_dir, "chat_dolly_15k.jsonl"))
    except Exception as e:
        print(f"❌ Error descargando Dolly: {e}")

    # 2. Alpaca Cleaned - El estándar de oro para seguir instrucciones
    print("\n--- Descargando Alpaca Cleaned ---")
    try:
        alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
        save_as_jsonl(alpaca, os.path.join(raw_dir, "chat_alpaca_cleaned.jsonl"))
    except Exception as e:
        print(f"❌ Error descargando Alpaca: {e}")

    # 3. OpenAssistant Guanaco (versión filtrada y limpia)
    print("\n--- Descargando OpenAssistant (Guanaco) ---")
    try:
        # Usamos una versión pre-procesada para no tener que lidiar con árboles de mensajes complejos
        oasst = load_dataset("timdettmers/openassistant-guanaco", split="train")
        save_as_jsonl(oasst, os.path.join(raw_dir, "chat_oasst_guanaco.jsonl"))
    except Exception as e:
        print(f"❌ Error descargando OpenAssistant: {e}")

    print("\n✅ Todos los datasets de chat han sido descargados en data/raw/")

if __name__ == "__main__":
    main()
