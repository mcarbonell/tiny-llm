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
    thought = f"ASSERT todos {g}s -> {p}. ASSERT {s} -> {g}. THEN {s} -> {p} [Law: Syllogism]."
    answer = f"{s} es {p}."
    return f"Pregunta: {prompt} <think> {thought} </think> Respuesta: {answer}"

def generate_math_logic():
    a = random.randint(1, 50)
    b = random.randint(1, 50)
    c = a + b
    prompt = f"Si tengo {a} naranjas y me regalan {b}, ¿cuántas tengo ahora?"
    thought = f"STEP 1: Identificar cantidad inicial ({a}). STEP 2: Identificar cantidad añadida ({b}). STEP 3: Operación SUMA [Law: Addition]. <calc>{a} + {b} = {c}</calc>."
    answer = f"Ahora tienes {c} naranjas."
    return f"Pregunta: {prompt} <think> {thought} </think> Respuesta: {answer}"

def generate_cyclic_logic():
    """Generador basado en V194-V195: Razonamiento de Módulo."""
    days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    start_idx = random.randint(0, 6)
    offset = random.randint(1, 14)
    target_idx = (start_idx + offset) % 7
    
    prompt = f"Si hoy es {days[start_idx]}, ¿qué día será dentro de {offset} días?"
    thought = f"Identificar ciclo: 7 días [Law: Modulus]. Calcular: ({start_idx} + {offset}) % 7 = {target_idx}. Mapear índice a día."
    answer = f"Será {days[target_idx]}."
    return f"Pregunta: {prompt} <think> {thought} </think> Respuesta: {answer}"

def generate_pattern_logic():
    """Generador de extrapolación de patrones (V190)."""
    elements = ["A", "B", "C", "X", "Y", "Z", "1", "2", "3"]
    pattern_len = random.randint(2, 3)
    pattern = [random.choice(elements) for _ in range(pattern_len)]
    full_seq = pattern * 4
    
    prompt = f"Completa la secuencia: {', '.join(full_seq[:-1])}, ..."
    thought = f"Detectar periodo: {pattern_len} [Law: Pattern Recognition]. El siguiente elemento debe ser {full_seq[-1]}."
    answer = f"El siguiente es {full_seq[-1]}."
    return f"Pregunta: {prompt} <think> {thought} </think> Respuesta: {answer}"

def main():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ Error: No se encuentra el tokenizador en {TOKENIZER_PATH}")
        return

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<eos>") or 0
    
    print("Generando 10,000 muestras de lógica avanzada (V195)...")
    samples = []
    generators = [generate_silogism, generate_math_logic, generate_cyclic_logic, generate_pattern_logic]
    
    for _ in range(10000):
        gen = random.choice(generators)
        samples.append(gen())
    
    print("\n--- MUESTRAS GENERADAS (EJEMPLOS) ---")
    for i in range(5):
        print(f"Muestra {i+1}: {samples[i]}")
    print("------------------------------------\n")
    
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
