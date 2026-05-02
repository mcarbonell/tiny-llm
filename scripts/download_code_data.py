"""
scripts/download_code_data.py
Descarga CodeAlpaca y Python Code Instructions para inyectar lógica de programación al Tokenizador.
"""
from datasets import load_dataset
import json
import os

def main():
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    # 1. CodeAlpaca - Instrucciones de código
    print("📥 Descargando CodeAlpaca-20k...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        output_path = os.path.join(raw_dir, "code_alpaca.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for e in ds:
                # CodeAlpaca has instruction, input, output
                instr = e.get("instruction", "")
                inp = e.get("input", "")
                out = e.get("output", "")
                if inp:
                    text = f"User: {instr}\nContext: {inp}\nAssistant: {out}"
                else:
                    text = f"User: {instr}\nAssistant: {out}"
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        print(f"✅ CodeAlpaca guardado ({os.path.getsize(output_path)} bytes)")
    except Exception as e:
        print(f"❌ Error CodeAlpaca: {e}")

    # 2. Python Pure - Lógica estructural
    print("📥 Descargando Python Code Instructions (18k)...")
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        output_path = os.path.join(raw_dir, "code_python_18k.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for e in ds:
                # Format: instruction, input, output
                text = f"User: {e['instruction']}\nAssistant: {e['output']}"
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        print(f"✅ Python Scripts guardado ({os.path.getsize(output_path)} bytes)")
    except Exception as e:
        print(f"❌ Error Python Scripts: {e}")
            
    print("\n🚀 Todos los datasets de código han sido descargados en data/raw/")

if __name__ == "__main__":
    main()
