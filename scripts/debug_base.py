import torch
import torch_directml
from model.model import TinyThinker, ModelArgs
import time

def debug_full_model():
    device = torch_directml.device()
    print(f"✅ Dispositivo: {device}")

    args = ModelArgs(dim=512, n_layers=12, n_heads=8, n_kv_heads=4, max_seq_len=512, vocab_size=16384)
    model = TinyThinker(args).to(device)

    x = torch.randint(0, 16384, (4, 512), device=device)
    y = torch.randint(0, 16384, (4, 512), device=device)

    print("\n--- Ejecutando Forward Completo ---")
    t0 = time.time()
    logits, loss = model(x, targets=y)
    
    print(f"Tiempo Forward: {time.time()-t0:.2f}s")
    print(f"¿Logits NaN?: {torch.isnan(logits).any().item()}")
    print(f"Valor del Loss: {loss.item()}")
    print(f"¿Loss NaN?: {torch.isnan(loss).item()}")

    print("\n--- Ejecutando Backward ---")
    t0 = time.time()
    loss.backward()
    print(f"Tiempo Backward: {time.time()-t0:.2f}s")

    nan_found = False
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            nan_found = True
            print(f"⚠️ NaN en gradiente de: {name}")
            break
            
    if not nan_found:
        print("✅ Gradientes limpios. El modelo es 100% estable en DirectML.")

if __name__ == "__main__":
    debug_full_model()
