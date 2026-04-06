import os
import sys
import re
import argparse
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

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "tokenizer.json")

def resolve_checkpoint(checkpoint_arg=None):
    if checkpoint_arg and os.path.exists(checkpoint_arg):
        return checkpoint_arg
    if checkpoint_arg:
        print(f"Warning: Checkpoint '{checkpoint_arg}' not found, falling back to auto-detect.")
    priority = ["ckpt_finetuned.pt", "ckpt_best.pt"]
    for name in priority:
        path = os.path.join(CHECKPOINTS_DIR, name)
        if os.path.exists(path):
            return path
    candidates = sorted([f for f in os.listdir(CHECKPOINTS_DIR) if f.startswith("ckpt_") and f.endswith(".pt")])
    if candidates:
        return os.path.join(CHECKPOINTS_DIR, candidates[-1])
    return None

def search_web_tool(query: str) -> str:
    """Implementación real de herramienta externa usando DuckDuckGo."""
    print(f"\n[TOOLCALL] Buscando en internet: '{query}'...", end="")
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if results:
            result_text = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
            print(" ¡Hecho!")
            return result_text
        else:
            print(" No results.")
            return "No se encontraron resultados relevantes."
    except Exception as e:
        print(f" Error: {e}. Usando fallback.")
        return f"Simulated search result for '{query}': This is a placeholder. Install duckduckgo-search for real results."

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
    previous_printed_text = ""
    
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
                query_str = re.sub(r'(?<=\w)\s(?=\w)', '', re.sub(r'\s+', ' ', tokenizer.decode(current_tool_query)).strip())
                
                # Ejecutar herramienta (El motor sale de PyTorch y vuelve a Python estándar)
                result = search_web_tool(query_str)
                
                # Inyectar resultado de vuelta a la mente de la red neuronal
                print(f"[INYECCIÓN AL CONTEXTO] <TOOL_RESULT> {result} </TOOL_RESULT>")
                tool_result_text = f" <TOOL_RESULT> {result} </TOOL_RESULT> "
                result_ids = tokenizer.encode(tool_result_text).ids
                
                # Anexar sin predecir
                x = torch.cat((x, torch.tensor([result_ids], device=DEVICE)), dim=1)
                
                print("\nTinyThinker> ", end="", flush=True)
                previous_printed_text = "" # Resetear el buffer de impresión
                generated_tokens = []      # Resetear buffer de tokens
            else:
                # Mientras genere la query, guardamos los IDs
                current_tool_query.append(token_id)
        else:
            # -------- FLUJO TEXTUAL NORMAL --------
            # Decodificar el token individual para evitar problemas de concatenación
            new_text = tokenizer.decode([token_id])
            print(new_text, end="", flush=True)
            
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Interactive chat with TinyThinker")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (auto-detects if not provided)")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    if ckpt_path is None:
        print("Error: No checkpoint found in checkpoints/ directory.")
        print("Run train.py first, or pass --checkpoint <path>.")
        sys.exit(1)

    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Tokenizer not found at {TOKENIZER_PATH}")
        print("Run download_and_tokenize.py first.")
        sys.exit(1)

    print(f"Using checkpoint: {os.path.basename(ckpt_path)}")
    print("Cargando Tokenizador...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    print("Despertando a TinyThinker desde el disco...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint['args']

    model = TinyThinker(model_args)
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
            generate_interactive(model, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)

        except KeyboardInterrupt:
            print("\nSaliendo de la terminal...")
            break

if __name__ == "__main__":
    main()
