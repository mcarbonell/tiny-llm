import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from model.model import TinyThinker, ModelArgs

def test_model_forward():
    # Instanciamos una versión de juguete de la red solo para comprobar cálculos de tensores
    args = ModelArgs(
        vocab_size=1000, 
        dim=128, 
        n_layers=2, 
        n_heads=4, 
        n_kv_heads=2,
        max_seq_len=64
    )
    model = TinyThinker(args)
    
    # Creamos un lote falso: Batch de 2, Secuencia de 16 tokens
    bsz = 2
    seqlen = 16
    tokens = torch.randint(0, args.vocab_size, (bsz, seqlen))
    
    # Forward Pass
    logits = model(tokens)
    
    # Tiene que devolver (Batch Size, Seq Len, Vocab Size)
    assert logits.shape == (bsz, seqlen, args.vocab_size), f"Error en las dimensiones devueltas: {logits.shape}"
    assert not torch.isnan(logits).any(), "Los logits contienen valores NaN (inestabilidad matemática)"
    print("Todo correcto: Las dimensiones del Forward Pass coinciden con la teoria.")

if __name__ == "__main__":
    test_model_forward()
