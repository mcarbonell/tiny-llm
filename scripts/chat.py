import os
import sys
import re
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Añadir ruta base para resolver import del modelo
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker, ModelArgs

# -----------------
# Configuración
# -----------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    DEVICE = 'mps'

CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")

def search_web_tool(query: str) -> str:
    """Implementación de la Herramienta externa simulada.
    En una versión de producción aquí inyectaríamos la API de Brave / Google / Wikipedia.
    """
    print(f"\n[TOOLCALL] Buscando en internet: '{query}'...", end="")
    try:
        # Mock de ejecución para validar el flujo
        import time
        time.sleep(1) # Simular latencia de red
        result = f"Los documentos recientes indican que '{query}' es importante."
        print(" ¡Hecho!")
        return result
    except Exception as e:
        print(f" Error: {e}")
        return "No se encontraron resultados."

def generate_interactive(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7, top_k=40):
    model.eval()
    
    # 1. Preparar IDs iniciales
    input_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    
    # 2. Identificadores de tokens especiales a cachear
    eos_id = tokenizer.token_to_id("<eos>") or tokenizer.token_to_id("<pad>")
    tool_call_id = tokenizer.token_to_id("<TOOL_CALL>")
    tool_call_end_id = tokenizer.token_to_id("</TOOL_CALL>")

    generated_tokens = []
    in_tool_call = False
    current_tool_query = []
    
    print("\nTinyThinker> ", end="", flush=True)

    for _ in range(max_new_tokens):
        # Acortar contexto para que quepa en la ventana del modelo local
        if x.size(1) > model.args.max_seq_len:
            x_cond = x[:, -model.args.max_seq_len:]
        else:
            x_cond = x
            
        with torch.no_grad():
            logits = model(x_cond)
            
        # Tomar los logits del último token predicho
        logits = logits[:, -1, :] / temperature
        
        # Filtro Top-K (elimina opciones extrañas)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Acumulación de contexto
        x = torch.cat((x, next_token), dim=1)
        token_id = next_token.item()
        
        # Salida prematura?
        if token_id == eos_id:
            break
            
        # -------- LOGICA DE HERRAMIENTAS (TOOL INTERCEPTION) --------
        if token_id == tool_call_id:
            in_tool_call = True
            current_tool_query = [] # Limpiamos buffer
            print("\n[DETECTADA INVOCACIÓN A HERRAMIENTA...]", end="", flush=True)
            continue
            
        if in_tool_call:
            if token_id == tool_call_end_id:
                in_tool_call = False
                # Reconstruir query
                query_str = tokenizer.decode(current_tool_query)
                query_str = query_str.strip()
                
                # Ejecutar herramienta (El motor sale de PyTorch y vuelve a Python estándar)
                result = search_web_tool(query_str)
                
                # Inyectar resultado de vuelta a la mente de la red neuronal
                print(f"[INYECCIÓN AL CONTEXTO] <TOOL_RESULT> {result} </TOOL_RESULT>")
                tool_result_text = f" <TOOL_RESULT> {result} </TOOL_RESULT> "
                result_ids = tokenizer.encode(tool_result_text).ids
                
                # Anexar sin predecir
                x = torch.cat((x, torch.tensor([result_ids], device=DEVICE)), dim=1)
                
                print("\nTinyThinker> ", end="", flush=True)
            else:
                # Mientras genere la query, guardamos los IDs
                current_tool_query.append(token_id)
        else:
            # -------- FLUJO TEXTUAL NORMAL --------
            # Es un token de texto estandar, lo imprimimos (esto mimetiza el stream de ChatGPT)
            text = tokenizer.decode([token_id])
            print(text, end="", flush=True)
            
    print("\n")

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"Error: No se encontró checkpoint en {CKPT_PATH}")
        print("Asegúrate de que el script train.py está guardando los pesos correctamente.")
        sys.exit(1)
        
    print("Cargando Tokenizador...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    print("Despertando a TinyThinker desde el disco...")
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    args = checkpoint['args']
    
    model = TinyThinker(args)
    # Ignorar warning si en el checkpoint no casan perfectos algunos diccionarios optativos
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    model.to(DEVICE)
    
    iter_n, loss_val = checkpoint.get('iter_num', 'N/A'), checkpoint.get('val_loss', 'N/A')
    if isinstance(loss_val, (int, float)):
        loss_val = f"{loss_val:.4f}"
    
    print(f"\n=============================================")
    print(f"¡Modelo Activo usando acelerador {DEVICE.upper()}!")
    print(f"Entrenamiento base -> Iteración: {iter_n} | Pérdida: {loss_val}")
    print("=============================================")
    print("Escribe tu mensaje ('exit' o 'quit' para salir).")
    
    while True:
        try:
            user_input = input("\nUsuario> ")
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ['exit', 'quit']:
                break
                
            prompt = f"User: {user_input}\nAssistant: "
            generate_interactive(model, tokenizer, prompt)
            
        except KeyboardInterrupt:
            print("\nSaliendo de la terminal...")
            break

if __name__ == "__main__":
    main()
