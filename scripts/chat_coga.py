import os
import sys
import torch
import torch.nn.functional as F

# Añadir ruta base para resolver import del modelo
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model_coga import TinyThinkerCOGA, ModelArgs

def simulate_coga_generation():
    print("==================================================")
    print(" SIMULADOR COGA - FASE 2 (Intercepción Simbólica) ")
    print("==================================================")
    
    # 1. Configurar modelo pequeño para simulación
    args = ModelArgs(dim=256, n_layers=2, vocab_size=1000, n_scratch_slots=32)
    model = TinyThinkerCOGA(args)
    model.eval()
    
    # 2. Definir IDs ficticios para nuestros tokens de control (Primitivas CRUD)
    # En un escenario real, estos se añadirían al tokenizer.json
    TOKEN_WRITE = 900
    TOKEN_END_WRITE = 901
    
    # 3. Estado inicial
    # Contexto simulado: "User: ¿Cuál es la capital de Francia? Assistant: "
    context_ids = [10, 25, 40, 15] 
    x = torch.tensor([context_ids], dtype=torch.long)
    
    # Inicializamos la RAM en blanco
    scratchpad = torch.zeros(1, args.n_scratch_slots, args.dim)
    
    print(f"\n[ESTADO INICIAL]")
    print(f"Longitud del contexto: {x.size(1)} tokens")
    print(f"Scratchpad: {scratchpad.shape} (Vacío)")
    
    # 4. Simulación de la salida del modelo
    # Imaginemos que el modelo decide pensar antes de responder
    simulated_model_output = [
        TOKEN_WRITE,  # El modelo decide usar el scratchpad
        105, 204,     # Piensa: "Francia -> París"
        TOKEN_END_WRITE, # Termina de pensar
        500, 600, 700 # Responde: "La capital es París"
    ]
    
    print("\n[INICIANDO GENERACIÓN...]")
    
    in_write_block = False
    thought_buffer = []
    output_text = []
    
    for step, token_id in enumerate(simulated_model_output):
        print(f"\n--- Paso {step + 1} | Modelo emite token_id: {token_id} ---")
        
        # Añadimos el token al contexto temporalmente
        x = torch.cat([x, torch.tensor([[token_id]], dtype=torch.long)], dim=1)
        
        if token_id == TOKEN_WRITE:
            print(">> [INTERCEPCIÓN] Token <WRITE> detectado. Entrando en Modo Pensamiento.")
            in_write_block = True
            thought_buffer = []
            continue
            
        if in_write_block:
            if token_id == TOKEN_END_WRITE:
                print(">> [INTERCEPCIÓN] Token <END_WRITE> detectado. Procesando pensamiento...")
                
                # A. Guardar en Scratchpad (Simulación)
                # En la realidad, pasaríamos thought_buffer por la capa de embedding del modelo
                # y lo guardaríamos en un slot libre.
                slot_to_use = 0
                thought_tensor = torch.randn(1, args.dim) # Embedding simulado del pensamiento
                scratchpad[0, slot_to_use] = thought_tensor
                print(f"   -> Pensamiento guardado en el SLOT {slot_to_use} de la RAM.")
                
                # B. LA MAGIA DE COGA: Borrar del contexto principal
                # Borramos el <WRITE>, los tokens de pensamiento, y el <END_WRITE>
                tokens_to_delete = len(thought_buffer) + 2 
                x = x[:, :-tokens_to_delete]
                print(f"   -> ¡MAGIA! Contexto rebobinado. Eliminados {tokens_to_delete} tokens de pensamiento de la ventana principal.")
                
                in_write_block = False
            else:
                thought_buffer.append(token_id)
                print(f"   (Pensando internamente: token {token_id})")
        else:
            # Flujo normal: el token es para el usuario
            output_text.append(str(token_id))
            print(f"Usuario ve: {token_id}")
            
    print("\n==================================================")
    print("[RESULTADO FINAL]")
    print(f"Respuesta visible al usuario: {' '.join(output_text)}")
    print(f"Longitud FINAL del contexto : {x.size(1)} tokens (¡Los pensamientos no consumieron espacio!)")
    print(f"Uso de la memoria RAM       : Slot 0 contiene datos, {args.n_scratch_slots - 1} slots libres.")
    print("==================================================")

if __name__ == "__main__":
    simulate_coga_generation()
