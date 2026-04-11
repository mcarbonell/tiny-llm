import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model_coga import TinyThinkerCOGA, ModelArgs

def test_phase4_recurrence():
    print("==================================================")
    print(" SIMULADOR COGA - FASE 4 (Profundidad Adaptativa) ")
    print("==================================================")
    
    args = ModelArgs(
        dim=256, 
        n_pre_layers=1, 
        n_core_layers=2, 
        n_post_layers=1, 
        max_recurrence_steps=5, # Máximo 5 iteraciones
        vocab_size=1000, 
        n_scratch_slots=32
    )
    model = TinyThinkerCOGA(args)
    model.eval() # Modo inferencia activa el paro adaptativo
    
    print(f"[ARQUITECTURA]")
    print(f"- Pre-layers: {args.n_pre_layers}")
    print(f"- Core-layers: {args.n_core_layers} (Recurrentes)")
    print(f"- Post-layers: {args.n_post_layers}")
    print(f"- Max Steps: {args.max_recurrence_steps}")
    
    # Input simulado (batch=1, seq_len=4)
    tokens = torch.randint(0, args.vocab_size, (1, 4))
    
    # Vamos a forzar la salida de la halt_head para simular
    # una tarea "fácil" y una "difícil".
    
    with torch.no_grad():
        # Anulamos los pesos para que el output sea exactamente igual al bias forzado
        model.halt_head.weight.data.zero_()
        
        print("\n[TEST A] Tarea 'FÁCIL' (Alto halt_prob)")
        # Forzamos un bias alto positivo -> sigmoid será ~1.0 -> 0 iteraciones extras (min 1)
        model.halt_head.bias.data = torch.tensor([10.0])
        
        # Inyectamos un hook para espiar el forward
        core_execution_count = 0
        def hook_fn(module, input, output):
            nonlocal core_execution_count
            core_execution_count += 1
            
        handle = model.core_layers[0].register_forward_hook(hook_fn)
        
        logits = model(tokens)
        print(f"-> Iteraciones reales del Core: {core_execution_count}")
        
        handle.remove()

        print("\n[TEST B] Tarea 'DIFÍCIL' (Bajo halt_prob)")
        # Forzamos un bias alto negativo -> sigmoid será ~0.0 -> max_recurrence_steps iteraciones
        model.halt_head.bias.data = torch.tensor([-10.0])
        
        core_execution_count = 0
        handle = model.core_layers[0].register_forward_hook(hook_fn)
        
        logits = model(tokens)
        print(f"-> Iteraciones reales del Core: {core_execution_count}")
        
        handle.remove()

    print("\n==================================================")
    if core_execution_count == args.max_recurrence_steps:
        print("EXITO: El modelo adapta su profundidad computacional basándose en la Halt Head.")
    else:
        print("FALLO: La recurrencia dinámica no funciona como se esperaba.")

if __name__ == "__main__":
    test_phase4_recurrence()
