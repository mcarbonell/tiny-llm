import torch
import time
from model.model import TinyThinker as TinyThinkerOriginal, ModelArgs
from model.model_flash import TinyThinker as TinyThinkerFlash

def test_parity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
    
    # Intentar usar DirectML si está disponible
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Using DirectML device: {device}")
    except ImportError:
        pass

    print(f"Testing parity on: {device}")

    args = ModelArgs(
        dim=256, 
        n_layers=4, 
        n_heads=8, 
        n_kv_heads=4, 
        max_seq_len=512
    )

    # 1. Inicializar modelos
    torch.manual_seed(42)
    model_orig = TinyThinkerOriginal(args).to(device)
    
    torch.manual_seed(42) # Misma semilla para que los pesos aleatorios sean idénticos
    model_flash = TinyThinkerFlash(args).to(device)

    # Verificar que los pesos son idénticos inicialmente
    for p1, p2 in zip(model_orig.parameters(), model_flash.parameters()):
        if not torch.equal(p1, p2):
            print("❌ Error: Los pesos iniciales no coinciden. Reintentando copia manual...")
            model_flash.load_state_dict(model_orig.state_dict())
            break

    model_orig.eval()
    model_flash.eval()

    # 2. Preparar input
    bsz, seqlen = 4, 128
    x = torch.randint(0, args.vocab_size, (bsz, seqlen)).to(device)

    # 3. Inferencia original
    t0 = time.time()
    with torch.no_grad():
        out_orig = model_orig(x)
        logits_orig = out_orig[0] if isinstance(out_orig, tuple) else out_orig
    t_orig = time.time() - t0

    # 4. Inferencia Flash
    t0 = time.time()
    with torch.no_grad():
        out_flash = model_flash(x)
        logits_flash = out_flash[0] if isinstance(out_flash, tuple) else out_flash
    t_flash = time.time() - t0

    # 5. Comparar resultados
    diff = torch.abs(logits_orig - logits_flash)
    max_diff = diff.max().item()
    avg_diff = diff.mean().item()

    print(f"\n" + "="*50)
    print(f"📊 RESULTADOS DEL TEST DE PARIDAD")
    print(f"="*50)
    print(f"Máxima diferencia: {max_diff:.2e}")
    print(f"Diferencia media:  {avg_diff:.2e}")
    print(f"Tiempo Original:   {t_orig:.4f}s")
    print(f"Tiempo Flash:      {t_flash:.4f}s")
    print(f"Mejora:            {(t_orig/t_flash - 1)*100:.1f}%")
    
    if max_diff < 1e-5:
        print(f"\n✅ TEST PASADO: Los modelos son matemáticamente equivalentes.")
    else:
        print(f"\n❌ TEST FALLIDO: Diferencia significativa detectada.")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_parity()
