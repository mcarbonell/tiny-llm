import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")
OUTPUT_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "wiki.bin")

def main():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<eos>") or 0

    # wikimedia/wikipedia, configuración "20231101.simple" = SimpleWiki
    dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)

    with open(OUTPUT_PATH, "wb") as f:
        buffer = []
        for i, example in enumerate(tqdm(dataset)):
            tokens = tokenizer.encode(example["text"]).ids
            tokens.append(eos_id)
            buffer.extend(tokens)
            if len(buffer) > 1_000_000:
                f.write(np.array(buffer, dtype=np.uint16).tobytes())
                buffer = []
        if buffer:
            f.write(np.array(buffer, dtype=np.uint16).tobytes())

    print(f"Wiki guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
