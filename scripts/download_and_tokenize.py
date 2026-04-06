import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ConfiguraciÃ³n
DATASET_NAME = "roneneldan/TinyStories"
VOCAB_SIZE = 16384 # Vocabulario pequeÃ±o para modelo ligero
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")

def main():
    print("Cargando el dataset TinyStories...")
    # Cargamos solo la particiÃ³n de train para entrenar el tokenizador.
    # El dataset completo tiene historias cortas generadas (~500M tokens en total),
    # usaremos un subconjunto (ej. el streaming o el primer 10-20%) para ser rÃ¡pidos.
    # De un vistazo, cargar todo en memoria puede tardar un poco, asÃ­ que lo manejaremos como un generador.
    dataset = load_dataset(DATASET_NAME, split='train', streaming=True)

    # Entrenar un Tokenizador BPE desde cero
    print("Inicializando Tokenizador BPE...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    # Definir tokens especiales
    special_tokens = ["<unk>", "<bos>", "<eos>", "<pad>", "<TOOL_CALL>", "</TOOL_CALL>", "<TOOL_RESULT>", "</TOOL_RESULT>", "<THINK>", "</THINK>", "[SYSTEM]", "[/SYSTEM]"]

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        show_progress=True
    )

    # Generador para alimentar al tokenizador sin saturar la RAM
    def batch_iterator(batch_size=1000, max_samples=500000):
        buffer = []
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
            buffer.append(example['text'])
            if len(buffer) == batch_size:
                yield buffer
                buffer = []
        if buffer:
            yield buffer

    # Entrenar (usaremos ~500k muestras para captar vocabulario)
    print("Entrenando el tokenizador (esto puede tardar unos minutos)...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Asegurarnos de que el directorio existe
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    
    # Guardar
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizador guardado exitosamente en: {TOKENIZER_PATH}")

if __name__ == "__main__":
    main()
