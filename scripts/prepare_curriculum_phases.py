import os
import numpy as np
import random
import argparse

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

def load_bin(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    return np.fromfile(path, dtype=np.uint16)

def create_phase(name, components, target_tokens=10_000_000):
    """
    components: list of tuples (array, weight, name)
    """
    print(f"\n--- Creating phase: {name} ---")
    
    mixed_data = []
    for data, weight, comp_name in components:
        if data is None:
            print(f"  [MISSING] {comp_name} (skipping)")
            continue
            
        n_tokens = int(target_tokens * weight)
        print(f"  [MIX] {comp_name}: {weight*100:.1f}% ({n_tokens:,} tokens)")
        
        # Random sampling for large datasets, repetition for small ones
        if len(data) > n_tokens:
            start = random.randint(0, len(data) - n_tokens)
            sub = data[start:start+n_tokens]
        else:
            # If logic data is scarce, we can repeat it or just take what we have
            # For logic, repetition is often okay in low doses
            repeats = (n_tokens // len(data)) + 1
            sub = np.tile(data, repeats)[:n_tokens]
            
        mixed_data.append(sub)
    
    if not mixed_data:
        print(f"  [ERROR] No data components available for {name}")
        return

    final_arr = np.concatenate(mixed_data)
    
    # Shuffle sequences (assuming sequences are roughly the same length)
    # This is a simple token-level shuffle which might break sequences 
    # if not handled by the trainer. Usually, we want to keep them but 
    # for pre-training mixture we often just save the stream.
    
    output_path = os.path.join(DATA_DIR, name)
    final_arr.tofile(output_path)
    print(f"  [DONE] Saved to {output_path} ({len(final_arr):,} tokens)")

def main():
    parser = argparse.ArgumentParser(description="Prepare training phases for TinyThinker Curriculum.")
    parser.add_argument("--tokens", type=int, default=5_000_000, help="Target tokens per phase")
    args = parser.parse_args()

    # Load Base Datasets
    stories = load_bin(os.path.join(DATA_DIR, "train.bin")) # TinyStories 
    wiki = load_bin(os.path.join(DATA_DIR, "wiki.bin"))       # SimpleWiki (if exists)

    # Load Logic Levels (L0 to L6)
    logic_levels = {}
    for i in range(7):
        path = os.path.join(DATA_DIR, f"logic_l{i}.bin")
        data = load_bin(path)
        if data is not None:
            logic_levels[i] = data

    # PHASE 1: Grammar & Narrative (Foundation)
    # Focus: Fluency and basic structure
    create_phase("phase1_grammar.bin", [
        (stories, 1.0, "TinyStories")
    ], target_tokens=args.tokens)

    # PHASE 2: Child Logic (Level 0 & 1)
    # Focus: Introducing logic within narrative context
    create_phase("phase2_child_logic.bin", [
        (stories, 0.70, "TinyStories"),
        (logic_levels.get(0), 0.15, "Logic L0"),
        (logic_levels.get(1), 0.15, "Logic L1")
    ], target_tokens=args.tokens)

    # PHASE 3: Structured Thinking (Level 2 & 3)
    # Focus: More complex inference and simple wiki facts
    create_phase("phase3_structured.bin", [
        (stories, 0.40, "TinyStories"),
        (wiki, 0.20, "Wiki"),
        (logic_levels.get(1), 0.10, "Logic L1 (Interleave)"),
        (logic_levels.get(2), 0.15, "Logic L2"),
        (logic_levels.get(3), 0.15, "Logic L3")
    ], target_tokens=args.tokens)

    # PHASE 4: Cognitive Peak (L4, L5, L6)
    # Focus: Formal logic, meta-reasoning, and advanced knowledge
    create_phase("phase4_expert.bin", [
        (wiki, 0.40, "Wiki"),
        (logic_levels.get(3), 0.10, "Logic L3 (Interleave)"),
        (logic_levels.get(4), 0.20, "Logic L4"),
        (logic_levels.get(5), 0.15, "Logic L5"),
        (logic_levels.get(6), 0.15, "Logic L6")
    ], target_tokens=args.tokens)

if __name__ == "__main__":
    main()
