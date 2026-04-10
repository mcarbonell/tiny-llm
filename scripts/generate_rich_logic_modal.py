import os
import requests
import numpy as np
import time
from tokenizers import Tokenizer
from tqdm import tqdm
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Modal (Gratuita para GLM-5.1)
API_URL = "https://api.us-west-2.modal.direct/v1/chat/completions"
# API_URL = "https://api.us-west-2.modal.direct/v1"
API_KEY = os.getenv("MODAL_API_KEY")
MODEL_NAME = "zai-org/GLM-5.1-FP8"

if not API_KEY:
    raise ValueError("❌ No se encontró MODAL_API_KEY en el archivo .env")

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_logic_rich.bin")

# Temas diversos para generar razonamiento variado
TOPICS = [
    "un rompecabezas lógico de deducción",
    "un caso de depuración de código Python con loops complejos",
    "un problema de planificación de tareas con restricciones de tiempo",
    "un silogismo aristotélico complejo",
    "una deducción sobre física intuitiva o sentido común",
    "un problema matemático de varios pasos (fracciones, porcentajes)",
    "un escenario hipotético de causa y efecto 'si ocurre A y no B...'",
    "explicar el estado de varias variables tras un bloque de código"
]

SYSTEM_PROMPT = """Eres un experto en generar datos de entrenamiento para modelos de pensamiento profundo (como DeepSeek-R1 o o1).
Tu misión es crear pares de PREGUNTA y RESPUESTA con un RAZONAMIENTO INTERNO extremadamente detallado y analítico.

DEBES seguir este formato exacto:
Pregunta: [La pregunta aquí]
<think> [Aquí el razonamiento paso a paso. Usa keywords como ASSERT (hechos), VERIFY (comprobaciones), BECAUSE (causas) y THEN (conclusiones). Sé muy estructurado. No uses markdown.] </think>
Respuesta: [La respuesta final, clara y directa]

IMPORTANTE: No te saludes ni añadidas nada fuera del formato. Razonamiento largo y denso en lógica."""

def get_rich_sample(topic, retry_count=0):
    """
    Función robusta con reintentos y backoff exponencial para manejar saturación de API.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Genera un ejemplo original sobre: {topic}. El bloque <think> debe ser extenso y muy lógico."}
        ],
        "max_tokens": 1200,
        "temperature": 0.8
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        # Timeout alto (90s) porque el modelo en Modal puede ser lento generando razonamiento largo
        response = requests.post(API_URL, json=payload, headers=headers, timeout=90)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        
        elif response.status_code == 429 or "concurrent" in response.text:
            # Backoff exponencial si hay demasiadas peticiones
            wait_time = min(2**retry_count + 10, 60)
            print(f"⚠️ API Saturada. Reintento {retry_count+1} en {wait_time}s...")
            time.sleep(wait_time)
            return get_rich_sample(topic, retry_count + 1)
        
        else:
            print(f"❌ Error API ({response.status_code}): {response.text}")
            return None
            
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        wait_time = 15
        print(f"⚠️ Timeout o error de conexión: {e}. Reintentando en {wait_time}s...")
        time.sleep(wait_time)
        return get_rich_sample(topic, retry_count + 1) if retry_count < 5 else None
    
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None

def main():
    if not os.path.exists(TOKENIZER_PATH):
        print("❌ Error: Necesitas el archivo model/tokenizer.json. Ejecuta los scripts previos.")
        return

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<eos>") or 0
    
    # Objetivo de generación
    n_samples = 400 
    print(f"🚀 Iniciando generación robusta de {n_samples} muestras con GLM-5.1...")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Contador de éxitos para esta sesión
    current_success = 0
    
    pbar = tqdm(total=n_samples)
    
    while current_success < n_samples:
        topic = TOPICS[current_success % len(TOPICS)]
        sample_text = get_rich_sample(topic)
        
        if sample_text and "<think>" in sample_text and "Respuesta:" in sample_text:
            # Formateamos para el modelo final
            full_text = f"[SYSTEM] Asistente de Razonamiento [/SYSTEM] {sample_text}"
            tokens = tokenizer.encode(full_text).ids
            tokens.append(eos_id)
            
            # GUARDADO INMEDIATO (Anexar al archivo .bin)
            # Esto permite reanudar el script simplemente volviéndolo a lanzar
            # (Anexa tokens uint16 al final del archivo existente)
            with open(OUTPUT_PATH, "ab") as f:
                f.write(np.array(tokens, dtype=np.uint16).tobytes())
            
            current_success += 1
            pbar.update(1)
            
            # Pequeña pausa de cortesía a los servidores de Modal
            time.sleep(3)
        else:
            # Si falla, esperamos un poco más antes de seguir con el bucle
            time.sleep(5)

    print(f"\n✅ Proceso completado exitosamente.")
    print(f"📁 Dataset enriquecido actualizado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
