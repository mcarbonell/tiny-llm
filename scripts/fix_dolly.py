from datasets import load_dataset
import json
import os

def fix_dolly():
    print("Fixing Dolly 15k...")
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        output_path = "data/raw/chat_dolly_15k.jsonl"
        
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                # Dolly has: instruction, context, response, category
                instr = entry.get("instruction", "")
                ctx = entry.get("context", "")
                resp = entry.get("response", "")
                
                if ctx:
                    text = f"User: {instr}\nContext: {ctx}\nAssistant: {resp}"
                else:
                    text = f"User: {instr}\nAssistant: {resp}"
                
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        
        print(f"✅ Success. Saved to {output_path} ({os.path.getsize(output_path)} bytes)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_dolly()
