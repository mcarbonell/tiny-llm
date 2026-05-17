import json
import random
import os
import string

# Plantillas de inyección de ruido y extracción de claves
TEMPLATES = [
    {
        "prefix": "[SYSTEM BOOT SEQ]\nALLOCATING_VRAM...\nKERNEL_PANIC_OVERRIDE_KEY = {key}\n[LOADING MODULES]\n",
        "noise_lines": "Loading module 0x{hex}: Status {status}\n",
        "suffix": "[FATAL ERROR] KERNEL PANIC. PLEASE PROVIDE OVERRIDE_KEY: {key}\n",
        "key_gen": lambda: "".join(random.choices(string.ascii_uppercase + string.digits, k=6)),
        "status_gen": lambda: random.choice(["OK", "WARN", "SKIP", "CORRUPTED"])
    },
    {
        "prefix": "--- CLASSIFIED DOSSIER ---\nAgent Name: {key}\nMission: Operation Silent Dawn\n--- BACKGROUND CHATTER ---\n",
        "noise_lines": "Intercepted comms (Sector {hex}): {status} anomaly detected.\n",
        "suffix": "--- END CHATTER ---\nConfirming identity of Operative for extraction: Agent {key}\n",
        "key_gen": lambda: random.choice(["Goliath", "Wraith", "Cipher", "Vanguard", "Specter", "Echo", "Odin", "Nova"]),
        "status_gen": lambda: random.choice(["Minor", "Severe", "Critical", "Negligible"])
    },
    {
        "prefix": "struct Config {{\n    int magic_seed = {key};\n}};\n// BEGIN COMPILED BYTECODE\n",
        "noise_lines": "0x{hex} : {status}\n",
        "suffix": "// END BYTECODE\n// To decrypt payload, re-enter magic_seed: {key}\n",
        "key_gen": lambda: str(random.randint(10000, 99999)),
        "status_gen": lambda: " ".join(random.choices(["MOV", "ADD", "JMP", "XOR", "PUSH", "POP", "CALL", "RET", "NOP"], k=4))
    }
]

def generate_random_hex():
    return "".join(random.choices(string.hexdigits.upper(), k=8))

def generate_sample(min_noise_lines=100, max_noise_lines=500):
    template = random.choice(TEMPLATES)
    key = template["key_gen"]()
    
    text = template["prefix"].format(key=key)
    
    num_lines = random.randint(min_noise_lines, max_noise_lines)
    for _ in range(num_lines):
        text += template["noise_lines"].format(
            hex=generate_random_hex(),
            status=template["status_gen"]()
        )
        
    text += template["suffix"].format(key=key)
    return text

def main():
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "hippocampus_stress_data.jsonl")
    
    num_samples = 10000 # ~10k muestras generan un corpus decente para stress test
    
    print(f"Generando {num_samples} muestras de stress-test para el Hipocampo...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Generamos longitudes de ruido variables para evitar que la red
            # se acostumbre a una "distancia" fija en el tiempo.
            sample_text = generate_sample(min_noise_lines=50, max_noise_lines=800)
            
            # Guardamos en formato JSONL estándar para posterior tokenización
            f.write(json.dumps({"text": sample_text}) + "\n")
            
            if (i + 1) % 1000 == 0:
                print(f"Generadas {i + 1} / {num_samples} muestras...")
                
    print(f"¡Dataset generado en {output_file}!")
    print("Para usarlo en entrenamiento, pásalo por el script de tokenización (ej: tokenize_rich_data.py)")

if __name__ == "__main__":
    main()
