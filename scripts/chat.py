import os
import sys
import re
import argparse
import logging
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Añadir ruta base para resolver import del modelo
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import TinyThinker, ModelArgs

# -----------------
# Configuración
# -----------------
DEVICE = 'cpu'
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"[device] DirectML activo: {DEVICE}")
except ImportError:
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        DEVICE = 'mps'
    print(f"[device] DirectML no disponible, usando: {DEVICE}")

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "tokenizer.json")

# Logger de módulo (se configura en main())
logger = logging.getLogger(__name__)

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
            logger.info(f"[search_web_tool] OK | query='{query}' | {len(results)} resultados")
            return result_text
        else:
            print(" No results.")
            logger.warning(f"[search_web_tool] Sin resultados | query='{query}'")
            return "No se encontraron resultados relevantes."
    except Exception as e:
        print(f" Error: {e}. Usando fallback.")
        logger.error(f"[search_web_tool] ERROR | query='{query}' | {type(e).__name__}: {e}")
        return f"Simulated search result for '{query}': This is a placeholder. Install duckduckgo-search for real results."

def generate_interactive(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7, top_k=40):
    """Genera texto con KV-cache para inferecia eficiente.
    [FIX BUG-D] Ahora usa past_key_values en lugar de recomputar toda la secuencia
    en cada paso. El prefijo se procesa una única vez (O(n)) y luego se genera
    token a token con los KV acumulados en O(1) por paso de atención.
    """
    model.eval()

    # 1. Preparar IDs iniciales
    input_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    # 2. Identificadores de tokens especiales
    eos_id = tokenizer.token_to_id("<eos>") or tokenizer.token_to_id("<pad>")
    tool_call_id = tokenizer.token_to_id("<TOOL_CALL>")
    tool_call_end_id = tokenizer.token_to_id("</TOOL_CALL>")

    in_tool_call = False
    current_tool_query = []
    past_key_values = None

    print("\nTinyThinker> ", end="", flush=True)

    with torch.no_grad():
        for step in range(max_new_tokens):
            # --- Primer paso: procesar prefijo completo ---
            # --- Pasos sucesivos: pasar solo el último token con KV-cache ---
            if step == 0:
                # Acortar contexto si es necesario
                x_cond = x[:, -model.args.max_seq_len:] if x.size(1) > model.args.max_seq_len else x
                logits, past_key_values = model(x_cond, use_cache=True)
            else:
                # Solo el último token generado: (1, 1)
                last_token = x[:, -1:]
                logits, past_key_values = model(last_token, past_key_values=past_key_values, use_cache=True)

            # Logits del último paso
            logits = logits[:, -1, :] / temperature

            # Filtro Top-K
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Añadir al contexto acumulado
            x = torch.cat((x, next_token), dim=1)
            token_id = next_token.item()

            # Salida prematura
            if token_id == eos_id:
                break

            # -------- LÓGICA DE HERRAMIENTAS (TOOL INTERCEPTION) --------
            if token_id == tool_call_id:
                in_tool_call = True
                current_tool_query = []
                print("\n[DETECTADA INVOCACIÓN A HERRAMIENTA...]", end="", flush=True)
                continue

            if in_tool_call:
                if token_id == tool_call_end_id:
                    in_tool_call = False
                    # Reconstruir query
                    query_str = re.sub(r'(?<=\w)\s(?=\w)', '', re.sub(r'\s+', ' ', tokenizer.decode(current_tool_query)).strip())

                    # Ejecutar herramienta
                    result = search_web_tool(query_str)

                    # Inyectar resultado al contexto
                    print(f"[INYECCIÓN AL CONTEXTO] <TOOL_RESULT> {result} </TOOL_RESULT>")
                    tool_result_text = f" <TOOL_RESULT> {result} </TOOL_RESULT> "
                    result_ids = tokenizer.encode(tool_result_text).ids
                    result_tensor = torch.tensor([result_ids], dtype=torch.long, device=DEVICE)

                    # Añadir tokens del resultado al contexto
                    x = torch.cat((x, result_tensor), dim=1)

                    # Resetear KV-cache para incluir el resultado inyectado
                    # Hay que re-codificar toda la secuencia actualizada
                    x_cond = x[:, -model.args.max_seq_len:] if x.size(1) > model.args.max_seq_len else x
                    logits, past_key_values = model(x_cond, use_cache=True)

                    print("\nTinyThinker> ", end="", flush=True)
                else:
                    current_tool_query.append(token_id)
            else:
                # -------- FLUJO TEXTUAL NORMAL --------
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

    # Configurar logging con FileHandler para registrar eventos del chat
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "chat.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )

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
