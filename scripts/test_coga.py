import torch
from model.model_coga import TinyThinkerCOGA, ModelArgs

def test_coga_architecture():
    print("--- Test 1: Inicialización de COGA (Fase 1 + Fase 2) ---")
    args = ModelArgs(
        dim=256,
        n_layers=2,
        n_experts=4,
        top_k=1,
        n_reserved=2,
        n_scratch_slots=8 # Usamos 8 slots para el test
    )
    model = TinyThinkerCOGA(args)
    print(f"Modelo COGA creado con {args.n_scratch_slots} slots de memoria RAM.")
    
    # Dummy inputs
    bsz = 2
    seqlen = 4
    tokens = torch.randint(0, args.vocab_size, (bsz, seqlen))
    
    print("\n--- Test 2: Forward sin Scratchpad (Auto-inicialización) ---")
    logits_no_scratch = model(tokens)
    if logits_no_scratch.shape == (bsz, seqlen, args.vocab_size):
        print("EXITO: Forward completado correctamente (Tensor de scratchpad auto-generado).")
    else:
        print(f"FALLO: Shape incorrecto {logits_no_scratch.shape}")

    print("\n--- Test 3: Flujo de Información desde el Scratchpad ---")
    # Para verificar que el modelo realmente usa el scratchpad, compararemos 
    # dos inferencias con el mismo contexto pero distinto contenido en RAM.
    
    # Scratchpad vacío
    scratchpad_empty = torch.zeros(bsz, args.n_scratch_slots, args.dim)
    logits_empty = model(tokens, scratchpad=scratchpad_empty)
    
    # Scratchpad con datos
    scratchpad_data = torch.randn(bsz, args.n_scratch_slots, args.dim)
    logits_data = model(tokens, scratchpad=scratchpad_data)
    
    # La diferencia en las salidas debería ser mayor a 0 si el modelo 
    # está prestando atención a la memoria RAM.
    diff = torch.abs(logits_empty - logits_data).sum().item()
    
    if diff > 1e-4:
        print(f"EXITO: La Cross-Attention funciona. La salida del modelo cambia cuando se altera el Scratchpad (Diferencia = {diff:.4f}).")
    else:
        print(f"FALLO: El modelo ignora el Scratchpad. Diferencia nula.")

if __name__ == "__main__":
    test_coga_architecture()
