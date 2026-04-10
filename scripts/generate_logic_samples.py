import os
import json
import random
from tokenizers import Tokenizer

# Configuración
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_logic.bin")

def generate_silogism():
    subjects = ["Sócrates", "Aristóteles", "Un perro", "Ese gato", "El monitor", "Un árbol"]
    groups = ["humano", "filósofo", "mamífero", "felino", "dispositivo electrónico", "planta"]
    properties = ["mortal", "sabio", "sangre caliente", "cazador", "necesita energía", "realiza fotosíntesis"]
    
    idx = random.randint(0, len(subjects)-1)
    s, g, p = subjects[idx], groups[idx], properties[idx]
    
    prompt = f"Si todos los {g}s son {p}es y {s} es un {g}, ¿qué se puede afirmar sobre {s}?"
    thought = f"ASSERT todos {g}s -> {p}. ASSERT {s} -> {g}. THEN {s} -> {p} (Silogismo categórico)."
    answer = f"{s} es {p}."
    return f"[SYSTEM] Eres un asistente lógico. [/SYSTEM] Pregunta: {prompt} <think> {thought} </think> Respuesta: {answer}"

def generate_math_logic():
    a = random.randint(1, 50)
    b = random.randint(1, 50)
    c = a + b
    prompt = f"Si tengo {a} naranjas y me regalan {b}, ¿cuántas tengo ahora?"
    thought = f"STEP 1: Identificar cantidad inicial ({a}). STEP 2: Identificar cantidad añadida ({b}). STEP 3: Operación SUMA. <calc>{a} + {b} = {c}</calc>. VERIFY resultado."
    answer = f"Ahora tienes {c} naranjas."
    return f"[SYSTEM] Asistente Matemático. [/SYSTEM] Pregunta: {prompt} <think> {thought} </think> {answer}"

def generate_coding_logic():
    var_name = random.choice(["x", "y", "contador", "total"])
    val = random.randint(0, 10)
    prompt = f"¿Qué hace este código: {var_name} = {val}; if ({var_name} > 5) print('Grande')?"
    thought = f"ASSERT {var_name} = {val}. VERIFY condicion {var_name} > 5. BECAUSE {val} {'es' if val > 5 else 'no es'} mayor que 5, THEN el código {'imprimirá' if val > 5 else 'no imprimirá'} 'Grande'."
    answer = f"El código imprimirá 'Grande'" if val > 5 else "El código no imprimirá nada."
    return f"[SYSTEM] Analista de Código. [/SYSTEM] Pregunta: {prompt} <think> {thought} </think> {answer}"

def main():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ Error: No se encuentra el tokenizador en {TOKENIZER_PATH}")
        return

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<eos>") or 0
    
    print("Generando 5000 muestras de lógica sintética...")
    samples = []
    generators = [generate_silogism, generate_math_logic, generate_coding_logic]
    
    for _ in range(5000):
        gen = random.choice(generators)
        samples.append(gen())
    
    # Tokenizar y guardar
    all_tokens = []
    for s in samples:
        all_tokens.extend(tokenizer.encode(s).ids)
        all_tokens.append(eos_id)
        
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    import numpy as np
    with open(OUTPUT_PATH, "wb") as f:
        f.write(np.array(all_tokens, dtype=np.uint16).tobytes())
        
    print(f"✅ Hecho. Muestras guardadas en {OUTPUT_PATH} ({len(all_tokens)} tokens)")

if __name__ == "__main__":
    main()
