import os
import json
import random
from openai import OpenAI

# Este script utiliza un LLM real ("Profesor") para generar un dataset estructurado.
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "tool_dataset_real.json")

# ==========================================
# CONFIGURACIÓN DEL LLM PROFESOR
# ==========================================
# Descomenta y ajusta el bloque que prefieras usar:

# --- Opción 1: OLLAMA (Local) ---
# client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# MODEL_NAME = "llama3" # Cambialo por el modelo que tengas (ej: phi3, mistral, llama3)

# --- Opción 2: LM STUDIO (Local) ---
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
MODEL_NAME = "google/gemma-4-26b-a4b"

# --- Opción 3: OPENROUTER (Cloud Gratuito) ---
# client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="SK-OR-TU-API-KEY")
# MODEL_NAME = "google/gemma-7b-it:free"

SYSTEM_PROMPT = """You are generating synthetic training data for a smaller AI model.
I will give you a question. You must reply IN THIS EXACT FORMAT, acting as a model that refuses to guess facts and uses external tools instead:

<THINK> [Briefly explain why you need to search] </THINK> <TOOL_CALL> search("[exact short query terms]") </TOOL_CALL> <TOOL_RESULT> [Invent a highly realistic search result snippet] </TOOL_RESULT> [Provide the final answer based ONLY on the result].

Example response:
<THINK> I don't store factual demographic data. I must search the web to be accurate. </THINK> <TOOL_CALL> search("Paris population 2023") </TOOL_CALL> <TOOL_RESULT> According to the 2023 census, the population of Paris is 2.1 million. </TOOL_RESULT> Based on the search, the population of Paris is 2.1 million.
"""

QUESTIONS = [
    "Who won the soccer world cup in 1998?",
    "What is the geographical deepest point in the ocean?",
    "Who wrote the book '1984'?",
    "What is the speed of light in vacuum?",
    "Who directed the movie Interstellar?",
    "What is the capital of Canada?",
    "When did the Roman Empire fall?",
    "How tall is Mount Everest in meters?"
]

def generate_real_example(query: str):
    """
    Envía la petición al LLM (Ollama/LM Studio) y recupera la traza.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=250
        )
        assistant_resp = response.choices[0].message.content.strip()
        
        # Validar rudimentariamente que el modelo ha seguido el formato
        if "<THINK>" not in assistant_resp or "<TOOL_CALL>" not in assistant_resp:
            return None
            
        # Ensamblaje final de la línea conversacional para TinyThinker
        full_text = f"User: {query}\nAssistant: {assistant_resp} <eos>"
        print(f"Generado correctamente: '{query[:30]}...'")
        return {"text": full_text}
    except Exception as e:
        print(f"Error conectando a la API: {e}. ¿Está Ollama/LMStudio ejecutándose?")
        return None

def main():
    print(f"Iniciando generación usando modelo profesor: '{MODEL_NAME}'")
    
    # Para usar la librería openai necesitas instalarla: `pip install openai`
    try:
        import openai
    except ImportError:
        print("Falta la librería 'openai'. Instálala con: pip install openai")
        return

    dataset = []
    
    # En un caso real iteraríamos sobre un archivo con 10,000 preguntas de QA genéricas
    # Aquí lo haremos x veces escupiendo Random questions.
    ITERATIONS = 50 
    
    for _ in range(ITERATIONS):
        q = random.choice(QUESTIONS)
        example = generate_real_example(q)
        if example:
            dataset.append(example)
            
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"\n¡Proceso finalizado! Se guardaron {len(dataset)} ejemplos reales en {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
