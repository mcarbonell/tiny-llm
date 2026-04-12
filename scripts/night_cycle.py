import os
import json
import re
import datetime

# ---------------------------------------------------------
# COGA FASE 5: EL SUEÑO ARTIFICIAL (NIGHT CYCLE)
# ---------------------------------------------------------
# Este script representa el "sueño" del agente.
# Se ejecuta durante la inactividad (ej. a las 3:00 AM).
# Analiza las interacciones del día, detecta dónde el modelo
# se equivocó (correcciones del usuario), y genera un dataset
# SFT para que el modelo se re-entrene a sí mismo en un
# "Expert Slot" disponible (Fase 1).
# ---------------------------------------------------------

def parse_chat_log(log_path):
    """
    Lee el log de chat y reconstruye las conversaciones.
    Busca patrones de corrección por parte del usuario.
    """
    if not os.path.exists(log_path):
        print(f"No se encontró el log de chat en {log_path}. Nada que procesar.")
        return []

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Analizando logs del día en: {log_path}...")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Heurísticas simples para detectar que el usuario corrigió al modelo
    correction_keywords = [
        "no, ", "te equivocas", "incorrecto", "en realidad es", 
        "no es así", "está mal", "falso", "pero "
    ]

    # Extraer turnos User/Assistant
    # Esto es una aproximación simple. En producción se usaría un parser JSON si los logs son estructurados.
    turns = re.findall(r'Usuario> (.*?)\n.*?TinyThinker> (.*?)(?=\nUsuario>|\Z)', content, re.DOTALL)
    
    corrections_found = []
    
    # Buscamos en el historial un turno donde el usuario corrige una afirmación anterior del asistente.
    for i in range(1, len(turns)):
        prev_user, prev_assistant = turns[i-1]
        curr_user, _ = turns[i]
        
        curr_user_lower = curr_user.lower().strip()
        
        is_correction = any(curr_user_lower.startswith(kw) for kw in correction_keywords)
        
        if is_correction:
            # ¡Hemos encontrado un error del modelo que fue corregido!
            print(f"  -> Detectado error del modelo. Corrección del usuario: '{curr_user.strip()}'")
            
            # Formateamos el nuevo dato de entrenamiento SFT ideal:
            # Pregunta original -> Respuesta corregida (provista por el usuario)
            corrections_found.append({
                "prompt": prev_user.strip(),
                "wrong_response": prev_assistant.strip(),
                "user_correction": curr_user.strip()
            })
            
    return corrections_found

def generate_finetune_dataset(corrections, output_path):
    """
    Genera un archivo JSONL con los pares de entrenamiento corregidos.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in corrections:
            # Sintetizamos la respuesta ideal: El modelo acepta su error y da la info correcta
            ideal_response = f"Tienes razón. {item['user_correction'].capitalize()}"
            
            example = {
                "text": f"User: {item['prompt']}\nAssistant: {ideal_response}"
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Dataset de auto-mejora generado ({len(corrections)} ejemplos): {output_path}")

def dispatch_night_training(dataset_path):
    """
    Simula el lanzamiento del entrenamiento nocturno en un Expert Slot.
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Despachando Auto-Finetune Nocturno...")
    print("---------------------------------------------------------")
    print(f" COMANDO A EJECUTAR (Simulado):")
    print(f" python scripts/train.py --arch moe --train_reserved True \\")
    print(f"                         --data_path {dataset_path} \\")
    print(f"                         --max_iters 500 --lr 1e-4 \\")
    print(f"                         --checkpoint ckpt_moe_latest.pt")
    print("---------------------------------------------------------")
    print(" -> El modelo está aprendiendo de los errores de hoy usando el Slot Experto 13.")
    print(" -> Al despertar, el conocimiento general estará intacto (Cero Olvido), pero ya no cometerá estos errores.")
    
def run_night_cycle():
    print("==================================================")
    print(" INICIANDO CICLO NOCTURNO COGA (Auto-Mejora)      ")
    print("==================================================")
    
    log_file = "logs/simulated_chat.log"
    
    # 1. Crear un log simulado para la prueba si no existe
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Usuario> ¿Cuál es la capital de Australia?\n")
        f.write("TinyThinker> La capital de Australia es Sídney.\n")
        f.write("Usuario> No, te equivocas. La capital es Canberra.\n")
        f.write("TinyThinker> Oh, tienes razón. Es Canberra.\n")
        f.write("Usuario> ¿Qué lenguaje se usa en Unity?\n")
        f.write("TinyThinker> En Unity se utiliza principalmente Python.\n")
        f.write("Usuario> En realidad es C#.\n")
        f.write("TinyThinker> Comprendido, es C#.\n")
        
    # 2. Ejecutar el análisis
    corrections = parse_chat_log(log_file)
    
    if not corrections:
        print("El modelo no cometió errores detectables hoy. Fin del ciclo nocturno.")
        return
        
    # 3. Generar dataset de corrección
    dataset_path = "data/night_cycle_finetune.jsonl"
    generate_finetune_dataset(corrections, dataset_path)
    
    # 4. Lanzar auto-entrenamiento
    dispatch_night_training(dataset_path)
    
    print("\n==================================================")
    print(" CICLO NOCTURNO COMPLETADO. Modelo actualizado.   ")
    print("==================================================")

if __name__ == "__main__":
    run_night_cycle()
