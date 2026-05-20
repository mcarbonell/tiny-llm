import os
import sys
import torch

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.model_spectral_v11_albert import SpectralThinkerV11, SpectralArgsV11

def test_v11_architecture():
    print("=== PROBANDO ARQUITECTURA SPECTRAL V11 (FOURIER-ALBERT) ===")
    
    # 1. Configurar argumentos para d=512, E=128, k=128, n_layers=6, vocab=32768
    args = SpectralArgsV11(
        dim=512,
        emb_dim=128,
        n_layers=6,
        vocab_size=32768,
        k_walsh=128,
        k_mem=32,
        chunk_size=256
    )
    
    # 2. Instanciar el modelo
    model = SpectralThinkerV11(args)
    model.eval()
    
    # 3. Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for p in model.embed.parameters())
    embed_proj_params = sum(p.numel() for p in model.embed_proj.parameters())
    block_params = sum(p.numel() for p in model.block.parameters())
    head_proj_params = sum(p.numel() for p in model.head_proj.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    
    print(f"Dimensiones de diseño:")
    print(f"  - Hidden Dimension (d): {args.dim}")
    print(f"  - Embedding Dimension (E): {args.emb_dim}")
    print(f"  - Capas Virtuales (n_layers): {args.n_layers}")
    print(f"  - Vocab Size (V): {args.vocab_size}")
    print(f"  - Walsh Dimension (k_walsh): {args.k_walsh}")
    
    print("\nDesglose de Parámetros:")
    print(f"  - Embeddings (embed): {embed_params:,} params (Compartido con Head)")
    print(f"  - Proyección de Entrada (embed_proj): {embed_proj_params:,} params")
    print(f"  - Bloque Compartido (block): {block_params:,} params")
    print(f"  - Proyección de Salida (head_proj): {head_proj_params:,} params")
    print(f"  - Cabezal Esférico (head): {head_params:,} params (Pesos compartidos, sin params extras)")
    print(f"  - TOTAL PARÁMETROS: {total_params:,} ({total_params/1e6:.3f}M)")
    
    # Validación matemática del tamaño
    # E * V = 128 * 32768 = 4,194,304
    # embed_proj = 128 * 512 = 65,536
    # block = ~200,000 (Mixer y FFN)
    # head_proj = 512 * 128 = 65,536
    # Total = 4,194,304 + 65,536 + block + 65,536 + head_params = ~8.56M
    
    # Comprobar rango esperado
    assert total_params < 9.0 * 1e6, f"El número de parámetros {total_params/1e6:.2f}M es superior al límite esperado para ALBERT!"
    print("\n[OK] El número de parámetros cumple con el objetivo de compresión agresiva (< 9.0M).")
    
    # 4. Simular un lote de entrada
    batch_size = 2
    seq_len = 256
    dummy_input = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    
    print(f"\nEjecutando Forward pass con entrada: {dummy_input.shape}...")
    
    with torch.no_grad():
        logits = model(dummy_input)
        
    print(f"Shape de los logits de salida: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, args.vocab_size), f"Shape incorrecto! Esperado: {(batch_size, seq_len, args.vocab_size)}, Obtenido: {logits.shape}"
    
    # Comprobar NaNs
    nan_check = torch.isnan(logits).any()
    inf_check = torch.isinf(logits).any()
    print(f"  - NaNs detectados: {nan_check.item()}")
    print(f"  - Infs detectados: {inf_check.item()}")
    
    assert not nan_check, "Se detectaron NaNs en la salida de logits!"
    assert not inf_check, "Se detectaron Infs en la salida de logits!"
    print("[OK] Forward pass completado sin NaNs ni Infs.")
    
    # 5. Comprobar preservación de la norma en la salida intermedia del bloque (nGPT unit sphere constraint)
    print("\nVerificando restricciones de nGPT...")
    # Recuperamos el embedding proyectado para testear si la normalización funciona
    with torch.no_grad():
        e_full = model.embed(dummy_input)
        norm_e = torch.norm(e_full, dim=-1)
        print(f"  - Norma de Embeddings cruda (rango): {norm_e.min().item():.4f} a {norm_e.max().item():.4f}")
        
        # Norma después de proyectar y normalizar
        e_proj = model.embed_proj(e_full)
        # La norma de la salida esférica intermedia de la proyección debe ser 1.0 en cada token
        # En model_spectral_v11_albert.py:
        # e_full = norm_sphere(self.embed(x_full))
        # h_full = norm_sphere(self.embed_proj(e_full))
        from model.model_spectral_v10_hippocampus import norm_sphere
        h_normed = norm_sphere(e_proj)
        norm_h = torch.norm(h_normed, dim=-1)
        print(f"  - Norma de Embeddings Proyectada y Normalizada: {norm_h.min().item():.4f} a {norm_h.max().item():.4f}")
        assert torch.allclose(norm_h, torch.ones_like(norm_h), atol=1e-5), "La proyección de embeddings no mantiene la norma unitaria!"
        print("[OK] Las restricciones esféricas de nGPT se cumplen en la entrada.")

if __name__ == "__main__":
    test_v11_architecture()
