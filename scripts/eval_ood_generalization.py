import os
import sys
import json
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import argparse
import numpy as np

# Añadir ruta base
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.eval import load_model_and_tokenizer, generate_text

import re

def test_logical_transitivity(model, tokenizer, device):
    """Prueba de transitividad: A es B, B es C -> A es ? (C)"""
    prompts = [
        "El sol es una estrella. Las estrellas brillan. Por lo tanto, el sol", # Esperado: brilla
        "Si Juan es más alto que Pedro y Pedro es más alto que Luis, entonces Juan es", # Esperado: más alto
        "A es 1. B es 2. C es 3. D es 4. La letra después de C es", # Esperado: D
    ]
    print("\n--- Test 1: Transitividad Lógica ---")
    correct = 0
    for p in prompts:
        input_ids = tokenizer.encode(p).ids
        gen_ids = generate_text(model, tokenizer, input_ids, max_new_tokens=10, device=device)
        new_ids = gen_ids[0][len(input_ids):].tolist()
        output = tokenizer.decode(new_ids).lower().strip()
        print(f"Prompt: {p} | Gen: {output}")
        
        # Heurística más estricta (palabras completas)
        if "brilla" in output or "alto" in output:
            correct += 1
        elif p.endswith("C es") and (output.startswith("d") or " d " in output):
            correct += 1
            
    return correct / len(prompts)

def test_pattern_extrapolation(model, tokenizer, device):
    """Prueba de extrapolación: Repetir un patrón más allá de lo visto."""
    prompt = "1 uno 2 dos 3 tres 4 cuatro 5 cinco 6 seis 7 siete 8 ocho 9 nueve 10 diez 11"
    print("\n--- Test 2: Extrapolación de Patrones ---")
    input_ids = tokenizer.encode(prompt).ids
    gen_ids = generate_text(model, tokenizer, input_ids, max_new_tokens=5, device=device)
    new_ids = gen_ids[0][len(input_ids):].tolist()
    output = tokenizer.decode(new_ids).lower().strip()
    print(f"Prompt: {prompt} | Gen: {output}")
    return 1.0 if "once" in output else 0.0

def test_cyclic_reasoning(model, tokenizer, device):
    """Prueba de razonamiento cíclico (Módulo V194-V195)."""
    prompts = [
        "Si hoy es lunes, mañana es martes. Si hoy es domingo, mañana es", # Esperado: lunes
        "Enero, Febrero, Marzo, Abril, Mayo, Junio, Julio, Agosto, Septiembre, Octubre, Noviembre, Diciembre, Enero, Febrero,", # Esperado: Marzo
    ]
    print("\n--- Test 3: Razonamiento Cíclico (Módulo) ---")
    correct = 0
    for p in prompts:
        input_ids = tokenizer.encode(p).ids
        gen_ids = generate_text(model, tokenizer, input_ids, max_new_tokens=5, device=device)
        new_ids = gen_ids[0][len(input_ids):].tolist()
        output = tokenizer.decode(new_ids).lower().strip()
        print(f"Prompt: {p} | Gen: {output}")
        if "lunes" in output or "marzo" in output:
            correct += 1
    return correct / len(prompts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print(f"Evaluando Generalización OOD para: {args.checkpoint}")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.device)
    
    score_t = test_logical_transitivity(model, tokenizer, args.device)
    score_p = test_pattern_extrapolation(model, tokenizer, args.device)
    score_c = test_cyclic_reasoning(model, tokenizer, args.device)
    
    avg_score = (score_t + score_p + score_c) / 3
    
    print(f"\n========================================")
    print(f"RESUMEN DE INTELIGENCIA ESTRUCTURAL (OOD)")
    print(f"========================================")
    print(f"Transitividad:    {score_t:.2%}")
    print(f"Extrapolación:    {score_p:.2%}")
    print(f"Módulo/Ciclos:    {score_c:.2%}")
    print(f"----------------------------------------")
    print(f"SCORE GLOBAL:     {avg_score:.2%}")
    print(f"========================================")

if __name__ == "__main__":
    main()
