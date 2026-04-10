import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import argparse

def tokenize_file(input_path, output_path, tokenizer_path):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<pad>") or 0

    print(f"Tokenizing {input_path} -> {output_path}...")
    
    # We use a large buffer to write in chunks
    buffer = []
    chunk_size = 1_000_000
    
    # We read the file in blocks of text (separated by \n\n)
    with open(input_path, "r", encoding="utf-8") as raw_f, open(output_path, "wb") as bin_f:
        # Simple split by double newline as we saved it
        content = raw_f.read()
        examples = content.split("\n\n")
        
        for example in tqdm(examples):
            if not example.strip(): continue
            
            tokens = tokenizer.encode(example).ids
            tokens.append(eos_id)
            buffer.extend(tokens)
            
            if len(buffer) >= chunk_size:
                bin_f.write(np.array(buffer, dtype=np.uint16).tobytes())
                buffer = []
        
        if buffer:
            bin_f.write(np.array(buffer, dtype=np.uint16).tobytes())

    print(f"✅ Tokenization complete. File saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generic dataset tokenizer")
    parser.add_argument("--input", required=True, help="Path to raw .txt file")
    parser.add_argument("--output", required=True, help="Path to output .bin file")
    parser.add_argument("--tokenizer", default="model/tokenizer.json", help="Path to tokenizer.json")
    
    args = parser.parse_args()
    
    # Resolve relative paths to absolute from the root of the project
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    input_abs = os.path.abspath(os.path.join(root_dir, args.input))
    output_abs = os.path.abspath(os.path.join(root_dir, args.output))
    tokenizer_abs = os.path.abspath(os.path.join(root_dir, args.tokenizer))
    
    tokenize_file(input_abs, output_abs, tokenizer_abs)

if __name__ == "__main__":
    main()
