import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

def main():
    dataset_name = "roneneldan/TinyStories"
    tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.bin")
    
    print("Cargando tokenizador...")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"No se encuentra el tokenizador en {tokenizer_path}. Ejecuta download_and_tokenize.py primero.")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Cargando dataset {dataset_name} en streaming...")
    # Usamos streaming para no saturar RAM y tomaremos ~200k historias para la Fase 1
    dataset = load_dataset(dataset_name, split='train', streaming=True)
    
    # TinyStories = historias muy cortas (promedio < 200 tokens)
    # Almacenaremos tokens como uint16 (ya que nuestro vocab_size es 16384)
    # Escribiremos directamente al fichero en pedazos para ahorrar memoria
    
    # ID especial de fin de texto (generalmente <eos>)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        # Fallback al final si no pre-entrenamos con un <eos> explícito en las strings
        eos_id = tokenizer.token_to_id("<pad>") or 0

    max_samples = 1000000 
    
    print(f"Tokenizando y guardando hasta {max_samples} muestras en {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Abrimos o creamos el binario para anexar bytes
    with open(output_path, "wb") as f:
        # Buffer en RAM antes de volcar a disco por velocidad
        buffer = []
        for i, example in enumerate(tqdm(dataset, total=max_samples)):
            if i >= max_samples:
                break
            
            # Tokenizar el texto
            tokens = tokenizer.encode(example["text"]).ids
            # Añadimos marca de fin de historia
            tokens.append(eos_id)
            buffer.extend(tokens)
            
            # Volcar cada 10,000 historias para no crecer infinitamente
            if len(buffer) > 1000000:
                # Convertir a numpy uint16 (soporta hasta 65,535) y volcar
                arr = np.array(buffer, dtype=np.uint16)
                f.write(arr.tobytes())
                buffer = []
                
        # Escribir lo sobrante
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            f.write(arr.tobytes())
            
    print(f"¡Hecho! Dataset pre-tokenizado guardado en: {output_path}")

if __name__ == "__main__":
    main()
