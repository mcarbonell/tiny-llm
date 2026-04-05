import os
import json
import random

# Este es un template para generar datos sintéticos "offline".
# Su objetivo real es generar miles de ejemplos con un LLM "Profesor" (ej. GPT-4 / Claude) 
# para enseñarle a nuestro pequeño TinyThinker cuándo y cómo usar la herramienta estricta.

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "tool_dataset.json")

# -----------------
# Batería de ejemplos pre-concebidos (Mock API)
# En el mundo real, aquí usaríamos una librería como `openai` (openai.ChatCompletion.create)
# pasándole un System Prompt para que actuara de generador de formatos estrictos.
# -----------------

FACTUAL_QUESTIONS = [
    "Who is the current president of France?",
    "What is the capital of Australia?",
    "When did the Apollo 11 moon landing happen?",
    "How tall is the Eiffel Tower?",
    "Who directed the movie Inception?"
]

def generate_mock_example(query: str):
    """
    Simula la generación de una traza perfecta para enseñar uso de herramientas.
    El LLM Profesor generaría este texto basándose en instrucciones de comportamiento.
    """
    
    # 1. El usuario pregunta algo que la red NO DEBE MEMORIZAR
    user_part = f"User: {query}\nAssistant: "
    
    # 2. El asistente "Pisa el freno" y razona estructuralmente
    think_part = "<THINK> I don't store factual data. I need to search the web to answer safely. </THINK> "
    
    # 3. Extrae la entidad a buscar y llama a la herramienta estrictamente
    tool_query = query.replace("?", "").replace("Who is ", "").replace("What is ", "").replace("When did ", "")
    tool_call_part = f"<TOOL_CALL> search(\"{tool_query.strip()}\") </TOOL_CALL>"
    
    # 4. El entorno (en fase de datos) inyecta el resultado fingido para que aprenda a leerlo
    tool_res_part = f" <TOOL_RESULT> According to Wiki, relevant search result for {tool_query} is available. </TOOL_RESULT> "
    
    # 5. La respuesta final amalgamada basándose SÓLO en la lectura anterior
    final_ans = f"Based on my search, the current factual information about {tool_query.strip()} is available."
    
    # Ensamblamos el bloque de entrenamiento puro.
    full_text = user_part + think_part + tool_call_part + tool_res_part + final_ans + " <eos>"
    return {"text": full_text}

def main():
    print("⏳ Iniciando generación de dataset de Herramientas + Chain of Thought...")
    
    dataset = []
    
    # Generar iterando
    for i in range(500):
        # En producción: batch_request a OpenAI
        q = random.choice(FACTUAL_QUESTIONS)
        example = generate_mock_example(q)
        dataset.append(example)
        if (i+1) % 100 == 0:
            print(f"✅ Generados {i+1} ejemplos sintéticos...")
            
    # Guardar en JSON (esto luego se tokenizaría a un archivo .bin, igual que con TinyStories)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"\n🚀 ¡Dataset sintético guardado en {OUTPUT_FILE} con {len(dataset)} ejemplos!")
    print("Nota: El archivo ha sido guardado puro en JSON. El siguiente paso en cadena de producción")
    print("sería pre-tokenizarlo y volver a arrancar train.py con él para el famoso 'Fine-Tuning'.")

if __name__ == "__main__":
    main()
