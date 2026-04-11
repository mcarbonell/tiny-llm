import torch
import torch.nn.functional as F

class MemoryBank:
    """
    Hipocampo de TinyThinker (COGA Fase 3).
    Actúa como una base de datos vectorial nativa utilizando la propia capa
    de embeddings del modelo para almacenar y recuperar recuerdos.
    """
    def __init__(self, dim):
        self.dim = dim
        self.memories = []
        self.embeddings = None # Tensor de dimensiones (N, dim)
        self.next_id = 0

    def get_text_embedding(self, text: str, model, tokenizer) -> torch.Tensor:
        """
        Genera un embedding denso promediando los embeddings de los tokens.
        En un modelo entrenado, esto captura la semántica de la frase.
        """
        # Encode con manejo de fallback para tests simulados
        if hasattr(tokenizer, 'encode'):
            ids = tokenizer.encode(text).ids
        else:
            ids = tokenizer(text) # fallback para mock tokenizers
            
        if not ids:
            ids = [0]
            
        device = model.tok_embeddings.weight.device
        x = torch.tensor([ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            # Obtenemos los embeddings puros de los tokens
            token_embs = model.tok_embeddings(x) # shape: (1, seq_len, dim)
            # Average pooling para obtener el embedding de la frase
            sentence_emb = token_embs.mean(dim=1) # shape: (1, dim)
            # Normalización L2 para similitud del coseno
            sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)
            
        return sentence_emb

    def remember(self, text: str, model, tokenizer, mem_type: str = "episodic") -> int:
        """
        Almacena un nuevo recuerdo en el banco de memoria a largo plazo.
        """
        emb = self.get_text_embedding(text, model, tokenizer)
        
        self.memories.append({
            "id": self.next_id,
            "text": text,
            "type": mem_type
        })
        
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = torch.cat([self.embeddings, emb], dim=0)
            
        mem_id = self.next_id
        self.next_id += 1
        return mem_id

    def recall(self, query_text: str, top_k: int, model, tokenizer) -> tuple:
        """
        Recupera los top_k recuerdos más relevantes para una query.
        Devuelve (lista_metadatos, tensor_de_embeddings).
        """
        if self.embeddings is None or len(self.memories) == 0:
            return [], None
            
        query_emb = self.get_text_embedding(query_text, model, tokenizer)
        
        # Similitud del Coseno (como los vectores están normalizados, es producto punto)
        similarities = torch.matmul(query_emb, self.embeddings.T).squeeze(0) # shape: (N,)
        
        k = min(top_k, similarities.size(0))
        values, indices = torch.topk(similarities, k)
        
        results = []
        retrieved_embeddings = []
        
        for i, idx in enumerate(indices.tolist()):
            mem = self.memories[idx]
            results.append({
                "id": mem["id"],
                "text": mem["text"],
                "score": values[i].item(),
                "type": mem["type"]
            })
            retrieved_embeddings.append(self.embeddings[idx].unsqueeze(0)) # shape: (1, dim)
            
        retrieved_tensor = torch.cat(retrieved_embeddings, dim=0) # shape: (k, dim)
        return results, retrieved_tensor

    def forget(self, memory_id: int) -> bool:
        """
        Elimina un recuerdo (olvido selectivo).
        """
        idx_to_remove = -1
        for i, mem in enumerate(self.memories):
            if mem["id"] == memory_id:
                idx_to_remove = i
                break
                
        if idx_to_remove == -1:
            return False
            
        self.memories.pop(idx_to_remove)
        
        # Actualizar tensor de embeddings
        if len(self.memories) == 0:
            self.embeddings = None
        else:
            self.embeddings = torch.cat([
                self.embeddings[:idx_to_remove], 
                self.embeddings[idx_to_remove+1:]
            ], dim=0)
            
        return True
