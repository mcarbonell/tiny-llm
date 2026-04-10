import os
from datasets import load_dataset
from tqdm import tqdm

def save_raw(name, dataset, limit=100000):
    """
    Downloads and saves a dataset as raw text in data/raw/
    """
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    raw_path = os.path.join(raw_dir, f"{name}.txt")
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"\nSaving {limit} samples to {raw_path}...")
    
    # We write line by line to be memory efficient
    with open(raw_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset, total=limit)):
            if i >= limit:
                break
            # Add story text and then a double newline as separator
            text = example["text"].strip()
            if text:
                f.write(text + "\n\n")

def main():
    # 1. TinyStories (Phase 1)
    print("--- Phase 1: TinyStories ---")
    try:
        ds_stories = load_dataset("roneneldan/TinyStories", split='train', streaming=True)
        save_raw("tinystories", ds_stories, limit=200000) # Save 200k stories as raw text
    except Exception as e:
        print(f"Error downloading TinyStories: {e}")

    # 2. Wikipedia Simple English (Phase 2)
    print("\n--- Phase 2: Simple Wikipedia ---")
    try:
        # Configuration "20231101.simple" is the standard for Simple English Wiki
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
        save_raw("wiki", ds_wiki, limit=50000) # Save 50k wiki articles as raw text
    except Exception as e:
        print(f"Error downloading Wikipedia: {e}")

    print("\n✅ Raw datasets downloaded to data/raw/")

if __name__ == "__main__":
    main()
