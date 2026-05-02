"""
scripts/generate_deep_poly_samples.py — Deep Polymorphism (V193)

Generador de datos sintéticos especializados en la COMPOSICIÓN DE LEYES.
Objetivo: Enseñar al modelo a resolver f(g(x)) combinando bancos analíticos.

Tipos de composición:
1. Matemática: Logaritmo de potencias, Seno de sumas.
2. Lógica: Transitividad condicional (A->B, B->C solo si D).
3. Cíclica: Ciclos anidados (Módulo sobre Módulo).
"""

import os
import json
import random
import math
from tokenizers import Tokenizer
import numpy as np

# Configuración
TOKENIZER_PATH = "model/tokenizer_v1.json"
OUTPUT_PATH = "data/synthetic_deep_poly.bin"

def generate_math_comp():
    """Composición funcional: f(g(x))"""
    x = random.randint(2, 10)
    # Operación interna g(x)
    g_val = x * x
    g_law = "[Law: Power]"
    
    # Operación externa f(g(x))
    # Vamos a usar una suma simple para g_val + offset
    offset = random.randint(1, 100)
    f_val = g_val + offset
    f_law = "[Law: Addition]"
    
    prompt = f"Calcula el resultado final si primero elevamos {x} al cuadrado y luego le sumamos {offset}."
    thought = f"STEP 1: Aplicar g(x) = {x}^2 = {g_val} {g_law}. STEP 2: Aplicar f(res) = {g_val} + {offset} = {f_val} {f_law}. Composición completada."
    answer = f"El resultado es {f_val}."
    return f"Pregunta: {prompt} <think> {thought} </think> Respuesta: {answer}"

def generate_logical_comp():
    """Lógica anidada / Transitividad Condicional."""
    # A -> B -> C (siempre que D sea azul)
    color = random.choice(["azul", "rojo"])
    prompt = f"Reglas: 1. Si el cielo es {color}, entonces A implica B. 2. B implica C. Si sabemos que el cielo es azul y A es verdadero, ¿qué podemos decir de C?"
    
    if color == "azul":
        thought = f"VERIFY condición cielo == azul. TRUE. APPLY transitividad A -> B -> C [Law: Transitivity]. Concluir C."
        answer = "C es verdadero."
    else:
        thought = f"VERIFY condición cielo == azul. FALSE (es rojo). La regla A -> B no se activa [Law: Conditional Logic]. No podemos asegurar B ni C."
        answer = "No se puede determinar."
        
    return f"Pregunta: {prompt} <think> {thought} </think> Respuesta: {answer}"

def generate_cyclic_comp():
    """Ciclos anidados: Módulo sobre Módulo."""
    # Un reloj que se adelanta en un ciclo semanal
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    dia_idx = random.randint(0, 6)
    saltos = random.randint(1, 10)
    
    prompt = f"Un guardia de seguridad rota de turno cada {saltos} días. Si hoy es {dias[dia_idx]}, ¿en qué día de la semana caerá su segundo turno desde hoy?"
    
    total_offset = saltos * 2
    final_idx = (dia_idx + total_offset) % 7
    
    thought = f"STEP 1: Calcular desplazamiento total (2 turnos * {saltos} días) = {total_offset} días [Law: Multiplication]. STEP 2: Aplicar ciclo semanal (módulo 7) sobre {dias[dia_idx]} [Law: Modulus]. ({dia_idx} + {total_offset}) % 7 = {final_idx}."
    answer = f"Su segundo turno será un {dias[final_idx]}."
    return f"Pregunta: {prompt} <think> {thought} </think> Respuesta: {answer}"

def main():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ Error: No se encuentra el tokenizador en {TOKENIZER_PATH}")
        return

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<eos>") or 0
    
    print("Generando 5,000 muestras de Composición de Leyes (V193)...")
    samples = []
    generators = [generate_math_comp, generate_logical_comp, generate_cyclic_comp]
    
    for _ in range(5000):
        gen = random.choice(generators)
        samples.append(gen())
        
    print("\n--- EJEMPLOS DE COMPOSICIÓN (V193) ---")
    for i in range(3):
        print(f"Ejemplo {i+1}: {samples[i]}")
    print("--------------------------------------\n")

    # Tokenizar y guardar
    all_tokens = []
    for s in samples:
        all_tokens.extend(tokenizer.encode(s).ids)
        all_tokens.append(eos_id)
        
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(np.array(all_tokens, dtype=np.uint16).tobytes())
        
    print(f"✅ Hecho. Muestras guardadas en {OUTPUT_PATH} ({len(all_tokens)} tokens)")

if __name__ == "__main__":
    main()
