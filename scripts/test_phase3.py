import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model_coga import TinyThinkerCOGA, ModelArgs
from model.memory import MemoryBank

class MockTokenizer:
    """Simulador de tokenizer para no depender del archivo real durante el test de arquitectura."""
    def encode(self, text):
        class Result:
            pass
        res = Result()
        # Genera IDs deterministas pero variados basados en las palabras
        res.ids = [(abs(hash(w)) % 1000) for w in text.split()]
        if not res.ids:
            res.ids = [0]
        return res

def test_phase3_memory():
    print("==================================================")
    print(" SIMULADOR COGA - FASE 3 (Memorias Programables)  ")
    print("==================================================")
    
    args = ModelArgs(dim=256, n_layers=2, vocab_size=1000, n_scratch_slots=32)
    model = TinyThinkerCOGA(args)
    model.eval()
    
    tokenizer = MockTokenizer()
    memory_bank = MemoryBank(dim=args.dim)
    
    print("\n[1] APRENDIZAJE: El modelo acumula experiencia en el Hipocampo...")
    m1 = memory_bank.remember("El usuario prefiere respuestas muy concisas.", model, tokenizer, "episodic")
    m2 = memory_bank.remember("La contraseña de la base de datos es 1234.", model, tokenizer, "semantic")
    m3 = memory_bank.remember("Python es el lenguaje favorito del usuario.", model, tokenizer, "episodic")
    
    print(f"-> Memorias almacenadas: {len(memory_bank.memories)}")
    
    print("\n[2] NUEVA INFERENCIA: Usuario hace una pregunta.")
    user_query = "¿En qué lenguaje debería escribir este script?"
    print(f"Query del usuario: '{user_query}'")
    
    # Simulamos el pre-procesamiento del Heartbeat o el hook de pre-inferencia
    print("\n[3] RECUPERACIÓN (Recall): Buscando recuerdos relevantes...")
    top_k = 2
    results, retrieved_embeddings = memory_bank.recall(user_query, top_k, model, tokenizer)
    
    for i, res in enumerate(results):
        print(f"   -> Top {i+1} (Score: {res['score']:.4f}): '{res['text']}' [{res['type']}]")
        
    print("\n[4] INYECCIÓN DE CONTEXTO AUMENTADO (RAG Nativo)")
    # ¡La magia de conectar Fase 3 con Fase 2!
    # El scratchpad tiene 32 slots. Ponemos las memorias a largo plazo en los primeros slots.
    bsz = 1
    scratchpad = torch.zeros(bsz, args.n_scratch_slots, args.dim)
    
    if retrieved_embeddings is not None:
        # retrieved_embeddings es (k, dim). Lo inyectamos en (1, k, dim)
        num_retrieved = retrieved_embeddings.size(0)
        scratchpad[0, :num_retrieved, :] = retrieved_embeddings
        print(f"-> ¡Éxito! Inyectadas {num_retrieved} memorias a largo plazo en los slots 0 a {num_retrieved-1} de la RAM editable (Scratchpad).")
        print(f"-> Slots restantes para pensar (Fase 2): {args.n_scratch_slots - num_retrieved}")
        
    print("\n[5] OLVIDO SELECTIVO: Borrando información...")
    print(f"El modelo decide olvidar la memoria ID {m2} (Contraseña)...")
    success = memory_bank.forget(m2)
    if success:
        print("-> Memoria borrada correctamente del Hipocampo.")
    else:
        print("-> Fallo al borrar.")
        
    print(f"-> Memorias restantes: {len(memory_bank.memories)}")
    print("==================================================")

if __name__ == "__main__":
    test_phase3_memory()
