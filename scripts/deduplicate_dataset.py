import os
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import re

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_question(text):
    # Extract everything before the first <think> tag
    match = re.search(r"(.*?)(?=<think>)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def deduplicate(input_file, threshold=0.92, dry_run=False, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"[*] Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Check for DirectML or CUDA if available, but default to CPU for safety in this script
    # unless the user has many samples.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model = model.to(device)
    
    print(f"[*] Reading dataset: {input_file}...")
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    
    if not samples:
        print("[!] Dataset is empty.")
        return

    questions = [extract_question(s['text']) for s in samples]
    
    print(f"[*] Encoding {len(questions)} questions...")
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(questions), batch_size)):
        batch = questions[i:i+batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings.cpu())
    
    all_embeddings = torch.cat(embeddings, dim=0)
    
    print("[*] Calculating similarity matrix...")
    # Similarity matrix: (N, N)
    sim_matrix = torch.mm(all_embeddings, all_embeddings.transpose(0, 1))
    
    to_remove = set()
    collisions = []
    
    print("[*] Finding collisions...")
    for i in range(len(samples)):
        if i in to_remove:
            continue
        
        # Find all highly similar samples
        similar_indices = torch.where(sim_matrix[i] > threshold)[0]
        
        for idx in similar_indices:
            idx = idx.item()
            if idx > i:  # Only look forward to avoid self-comparison and double counting
                to_remove.add(idx)
                collisions.append({
                    "original_idx": i,
                    "duplicate_idx": idx,
                    "similarity": sim_matrix[i][idx].item(),
                    "text_a": questions[i][:100] + "...",
                    "text_b": questions[idx][:100] + "..."
                })

    print(f"\n[REPORT] Found {len(collisions)} potential duplicates out of {len(samples)} samples.")
    
    if collisions:
        print("\n--- COLLISION PREVIEW ---")
        for c in collisions[:10]: # Show first 10
            print(f"- [{c['similarity']:.4f}] Sample {c['original_idx']} <-> {c['duplicate_idx']}")
            print(f"  A: {c['text_a']}")
            print(f"  B: {c['text_b']}")
        if len(collisions) > 10:
            print(f"... and {len(collisions)-10} more.")

    if not dry_run:
        output_file = input_file.replace(".jsonl", "_dedup.jsonl")
        print(f"\n[*] Saving sanitized dataset to: {output_file}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i, sample in enumerate(samples):
                if i not in to_remove:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"[SUCCESS] Saved {len(samples) - len(to_remove)} unique samples.")
    else:
        print("\n[DRY RUN] No file was created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate a JSONL dataset using semantic embeddings.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--threshold", type=float, default=0.92, help="Similarity threshold (0-1)")
    parser.add_argument("--dry-run", action="store_true", help="Only show the report")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace model name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[!] File not found: {args.input}")
    else:
        deduplicate(args.input, threshold=args.threshold, dry_run=args.dry_run, model_name=args.model)
