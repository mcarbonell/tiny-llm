import os
import json
import numpy as np
import argparse
from tokenizers import Tokenizer
from tqdm import tqdm

# Constants
DEFAULT_TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")

def main():
    parser = argparse.ArgumentParser(description="Tokenize rich reasoning JSONL data to binary.")
    parser.add_argument("--input", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "logic_l0.jsonl"), help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output BIN file")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH, help="Path to tokenizer.json")
    
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found at {args.tokenizer}")
        return
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<eos>") or 0
    
    all_tokens = []
    print(f"Tokenizing {args.input}...")
    
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        if not line.strip(): continue
        try:
            data = json.loads(line)
            # System prefix to guide the model's mode (Reasoning)
            text = f"[SYSTEM] Reasoning Engine [/SYSTEM] {data['text']}"
            tokens = tokenizer.encode(text).ids
            tokens.append(eos_id)
            all_tokens.extend(tokens)
        except Exception as e:
            print(f"Error processing line: {e}")
            continue
            
    # Convert to binary
    if not all_tokens:
        print("No tokens generated. Check your input file.")
        return

    print(f"Saving {len(all_tokens):,} tokens to {args.output}...")
    np.array(all_tokens, dtype=np.uint16).tofile(args.output)
    print(f"Done!")

if __name__ == "__main__":
    main()
