import json
import os
from datasets import load_dataset
import random

# CONFIGURACIÓN
GOLDEN_PATH = "data/dataset_golden_v1.json"
TOOL_DATA_RAW = "data/tool_dataset_real.json"
SYSTEM_TEXT = "[SYSTEM] You are TinyThinker, a compact AI assistant. You cannot reliably recall specific facts or dates. When asked factual questions, use your search tool. [/SYSTEM]"

def clean_tool_text(text):
    import re
    # Quitar los <thought> internos que generó Gemini si los hay
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
    # Asegurar que tiene el [SYSTEM] correcto
    if not text.startswith("[SYSTEM]"):
        text = f"{SYSTEM_TEXT}\n{text}"
    return text

def main():
    print("🚀 Iniciando construcción del Dataset Dorado...")
    
    final_dataset = []

    # 1. CARGAR DATOS DE HERRAMIENTAS (500 únicos)
    if os.path.exists(TOOL_DATA_RAW):
        with open(TOOL_DATA_RAW, "r", encoding="utf-8") as f:
            tool_data = json.load(f)
        for item in tool_data:
            item['text'] = clean_tool_text(item['text'])
            final_dataset.append(item)
        print(f"  ✅ Añadidos {len(tool_data)} ejemplos de Tool-Calling.")

    # 2. CARGAR DOLLY-15K PARA INSTRUCCIONES REALES (300 únicos)
    print("  📥 Descargando muestra de Dolly-15K...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    
    # Categorías que nos interesan para TinyThinker
    categories = {
        "creative_writing": "The user wants a creative response. I will use my narrative skills.",
        "brainstorming": "The user needs ideas. I will provide a structured list.",
        "general_qa": "This is a general question. I will provide a clear answer.",
        "classification": "Classification task. I will categorize the information accurately."
    }
    
    count = 0
    for example in dolly:
        cat = example["category"]
        if cat in categories:
            instr = example["instruction"]
            resp = example["response"]
            think = categories[cat]
            
            full_text = f"{SYSTEM_TEXT}\nUser: {instr}\nAssistant: <THINK> {think} </THINK> {resp} <eos>"
            final_dataset.append({"text": full_text})
            count += 1
        
        if count >= 300: break
    print(f"  ✅ Añadidos {count} ejemplos de Instrucciones Reales (Dolly).")

    # 3. AÑADIR CONVERSACIÓN CASUAL Y LÓGICA (Manual / Únicos)
    casual_chat = [
        {"instr": "Hello!", "think": "Greeting.", "resp": "Hi! I am TinyThinker. How can I help you today?"},
        {"instr": "How are you?", "think": "Polite inquiry.", "resp": "I am functioning perfectly. Ready to help you!"},
        {"instr": "What is your name?", "think": "Self-identity.", "resp": "My name is TinyThinker, a compact but smart AI model."},
        {"instr": "Tell me a joke.", "think": "Humor.", "resp": "Why did the computer go to the doctor? Because it had a virus!"},
        {"instr": "What is 10 + 5?", "think": "Simple math.", "resp": "10 + 5 is 15."},
        {"instr": "If I have 2 apples and you give me 3, how many do I have?", "think": "Logical addition.", "resp": "You would have 5 apples."},
    ]
    
    for c in casual_chat:
        full_text = f"{SYSTEM_TEXT}\nUser: {c['instr']}\nAssistant: <THINK> {c['think']} </THINK> {c['resp']} <eos>"
        final_dataset.append({"text": full_text})
    print(f"  ✅ Añadidos {len(casual_chat)} ejemplos de Conversación Casual.")

    # 4. MEZCLAR Y GUARDAR
    random.shuffle(final_dataset)
    
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)
    
    print(f"\n✨ ¡Dataset Dorado creado con éxito en: {GOLDEN_PATH}!")
    print(f"📊 Total de ejemplos únicos: {len(final_dataset)}")

if __name__ == "__main__":
    main()
