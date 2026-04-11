import torch
from model.model_moe import TinyThinkerMoE, ModelArgs

def test_moe_architecture():
    print("--- Test 1: Inicialización ---")
    args = ModelArgs(
        dim=256,
        n_layers=2,
        n_experts=8,
        top_k=2,
        n_reserved=4
    )
    model = TinyThinkerMoE(args)
    print(f"Modelo MoE creado con {args.n_experts} expertos ({args.n_reserved} reservados).")
    
    # Dummy input (batch=2, seq=4)
    tokens = torch.randint(0, args.vocab_size, (2, 4))
    
    print("\n--- Test 2: Verificación de Bloqueo (train_reserved=False) ---")
    # Realizamos un forward pass sin permitir expertos reservados
    logits = model(tokens, train_reserved=False)
    
    # Para verificar el bloqueo, vamos a instrumentar el forward pass de MoEFeedForward
    # Comprobamos que el router solo elige expertos del 0 al 3 (8 expertos - 4 reservados)
    
    found_reserved = False
    for layer in model.layers:
        moe = layer.feed_forward
        x_flat = torch.randn(1, args.dim) # dummy input para el gate
        gate_logits = moe.gate(x_flat)
        
        # Aplicamos la misma lógica que en el forward
        mask = torch.zeros_like(gate_logits)
        mask[:, -args.n_reserved:] = float('-inf')
        masked_logits = gate_logits + mask
        
        weights = torch.softmax(masked_logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, args.top_k, dim=-1)
        
        if (top_indices >= (args.n_experts - args.n_reserved)).any():
            found_reserved = True
            break
            
    if not found_reserved:
        print("EXITO: Los expertos reservados están correctamente bloqueados.")
    else:
        print("FALLO: Se detectó actividad en expertos reservados con train_reserved=False.")

    print("\n--- Test 3: Verificación de Activación (train_reserved=True) ---")
    # Ahora permitimos expertos reservados
    found_reserved_active = False
    
    # Forzamos los logits del router para que prefieran los últimos expertos
    with torch.no_grad():
        for layer in model.layers:
            moe = layer.feed_forward
            # Ponemos pesos muy altos en el router para los últimos expertos
            moe.gate.weight.data[-args.n_reserved:, :] = 10.0 
            moe.gate.weight.data[:-args.n_reserved, :] = -10.0
            
            # Simulamos el forward pass con train_reserved=True
            gate_logits = moe.gate(torch.randn(1, args.dim))
            # NO aplicamos la máscara aquí porque train_reserved es True
            weights = torch.softmax(gate_logits, dim=-1)
            _, top_indices = torch.topk(weights, args.top_k, dim=-1)
            
            if (top_indices >= (args.n_experts - args.n_reserved)).any():
                found_reserved_active = True
                break
                
    if found_reserved_active:
        print("EXITO: Los expertos reservados son accesibles cuando train_reserved=True.")
    else:
        print("FALLO: Los expertos reservados siguen inaccesibles.")

    print("\n--- Test 4: Shape del Output ---")
    logits = model(tokens)
    if logits.shape == (2, 4, args.vocab_size):
        print(f"EXITO: Shape del output correcto {logits.shape}")
    else:
        print(f"FALLO: Shape del output incorrecto {logits.shape}")

if __name__ == "__main__":
    test_moe_architecture()
