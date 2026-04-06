import os
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

# Carga variables desde el archivo .env
load_dotenv()

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
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
# MODEL_NAME = "google/gemma-4-26b-a4b"

# --- Opción 3: OPENROUTER (Cloud Gratuito) ---
# client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="SK-OR-TU-API-KEY")
# MODEL_NAME = "google/gemma-7b-it:free"

# --- Opción 4: GOOGLE AI STUDIO (Gemini API - Ultra veloz) ---
# Usa tu API key gratuita de Google AI Studio. 
# La librería 'openai' es 100% compatible con Google Gemini si apuntas a su base_url.
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
    api_key=os.getenv("GEMINI_API_KEY")
)
MODEL_NAME = "gemma-4-31b-it"

# Si usas APIs gratuitas con Rate Limits (ej. 15 peticiones por minuto de Google), activa esto:
RATE_LIMIT_DELAY = 4.5 # Segundos de espera entre pregunta y pregunta (15 RPM)

SYSTEM_PROMPT = """You are generating synthetic training data for a smaller AI model called TinyThinker.

The training format is:
[SYSTEM] You are TinyThinker, a compact AI assistant. You cannot reliably recall specific facts or dates. When asked factual questions, use your search tool. [/SYSTEM]
User: {question}
Assistant: <THINK> [brief reasoning] </THINK> <TOOL_CALL> search("query") </TOOL_CALL> <TOOL_RESULT> [realistic result] </TOOL_RESULT> [final answer]

I will give you a question. Reply ONLY with the Assistant turn (starting from <THINK>).
"""

QUESTIONS = [
    "Who won the soccer world cup in 1998?",
    "What is the geographical deepest point in the ocean?",
    "Who wrote the book '1984'?",
    "What is the speed of light in vacuum?",
    "Who directed the movie Interstellar?",
    "What is the capital of Canada?",
    "When did the Roman Empire fall?",
    "How tall is Mount Everest in meters?",
    "What is the population of Tokyo?",
    "Who painted the Mona Lisa?",
    "How far is the moon from Earth?",
    "What is the chemical formula for water?",
    "Who invented the telephone?",
    "What year did World War II end?",
    "Which planet is closest to the sun?",
    "What is the largest mammal on earth?",
    "Who discovered penicillin?",
    "What is the hardest natural substance?",
    "When was the Declaration of Independence signed?",
    "Who was the first president of the United States?",
    "What is the currency of Japan?",
    "Which ocean is the largest?",
    "Who wrote the play Romeo and Juliet?",
    "What temperature does water boil at in Celsius?",
    "What is the capital of Australia?",
    "Who is the CEO of Tesla?",
    "What is the chemical symbol for Gold?",
    "How many bones are in the adult human body?",
    "Who developed the theory of relativity?",
    "What is the tallest building in the world?",
    "When did the Apollo 11 moon landing happen?",
    "What is the square root of 144?",
    "Who composed the Four Seasons?",
    "What is the capital of Brazil?",
    "How many planets are in our solar system?",
    "Who wrote the Harry Potter series?",
    "What is the smallest country in the world?",
    "Which element has the atomic number 1?",
    "Who was the first person to walk on the moon?",
    "What is the longest river in the world?"
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
            max_tokens=500
        )
        assistant_resp = response.choices[0].message.content.strip()
        
        # Validar rudimentariamente que el modelo ha seguido el formato
        if "<THINK>" not in assistant_resp or "<TOOL_CALL>" not in assistant_resp:
            return None
            
        # Ensamblaje final de la línea conversacional para TinyThinker
        SYSTEM_TEXT = "[SYSTEM] You are TinyThinker, a compact AI assistant. You cannot reliably recall specific facts or dates. When asked factual questions, use your search tool. [/SYSTEM]"
        full_text = f"{SYSTEM_TEXT}\nUser: {query}\nAssistant: {assistant_resp} <eos>"
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
    ITERATIONS = 500 
    
    for _ in range(ITERATIONS):
        q = random.choice(QUESTIONS)
        example = generate_real_example(q)
        if example:
            dataset.append(example)
            
        import time
        if RATE_LIMIT_DELAY > 0:
            time.sleep(RATE_LIMIT_DELAY)
            
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"\n¡Proceso finalizado! Se guardaron {len(dataset)} ejemplos reales en {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
