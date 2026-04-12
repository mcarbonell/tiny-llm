import json
import os
import random

def generate_math_problem():
    """Genera problemas simples de aritmética de 2 dígitos."""
    op = random.choice(['+', '-', '*'])
    a = random.randint(10, 99)
    
    if op == '+':
        b = random.randint(10, 99)
        ans = a + b
        thought = f"Suma por partes: {a//10*10} + {b//10*10} = {(a//10*10)+(b//10*10)}. Unidades: {a%10} + {b%10} = {(a%10)+(b%10)}. Total: {(a//10*10)+(b//10*10)} + {(a%10)+(b%10)} = {ans}."
    elif op == '-':
        b = random.randint(10, a) # Evitar negativos para hacerlo simple
        ans = a - b
        thought = f"Resta de decenas: {a//10*10} - {b//10*10} = {(a//10*10)-(b//10*10)}. Resta de unidades: {a%10} - {b%10} = {(a%10)-(b%10)}. Total: {ans}."
    elif op == '*':
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        ans = a * b
        thought = f"Multiplicar por partes: {a} * {b//10*10} = {a*(b//10*10)}. {a} * {b%10} = {a*(b%10)}. Suma: {a*(b//10*10)} + {a*(b%10)} = {ans}."

    return a, op, b, ans, thought

def generate_dataset(num_samples: int, output_file: str):
    """
    Genera un dataset enseñando al modelo a usar las etiquetas del Scratchpad
    para resolver problemas matemáticos antes de emitir la respuesta final.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            # Tarea: Matemáticas
            a, op, b, ans, thought = generate_math_problem()
            
            # Formato ChatML adaptado con COGA Scratchpad primitives
            prompt = f"User: ¿Cuánto es {a} {op} {b}?\nAssistant: "
            
            # El truco: La respuesta ideal incluye el bloque de pensamiento rodeado
            # de los tokens de control, y solo después la respuesta visible al usuario.
            response = f"<WRITE>{thought}<END_WRITE>El resultado es {ans}."
            
            # Guardar en formato jsonl
            example = {
                "text": prompt + response
            }
            
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    out_path = "data/synthetic_coga_math.jsonl"
    print(f"Generando dataset COGA (Scratchpad Trace-based) en: {out_path}")
    generate_dataset(2000, out_path)
    print("¡Generación completada!")
    
    # Mostrar un ejemplo
    print("\nEjemplo de dato generado (Ground Truth):")
    with open(out_path, "r", encoding="utf-8") as f:
        print(json.loads(f.readline())['text'])
