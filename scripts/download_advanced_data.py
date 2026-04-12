import os
from datasets import load_dataset
from tqdm import tqdm

def download_and_save(repo_id, config_name, output_file, num_samples=100000):
    """
    Descarga una muestra de un dataset de Hugging Face y lo guarda en formato TXT local.
    """
    print(f"\n--- Descargando muestra de {repo_id} ({config_name}) ---")
    
    # Cargamos el dataset en modo streaming para no llenar la RAM
    ds = load_dataset(repo_id, config_name, split="train", streaming=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=num_samples, desc=f"Guardando en {os.path.basename(output_file)}")
        for entry in ds:
            text = entry.get("text", "")
            if text:
                f.write(text + "\n<|endoftext|>\n")
                count += 1
                pbar.update(1)
            
            if count >= num_samples:
                break
        pbar.close()
    
    print(f"✅ Completado: {count} documentos guardados.")

def main():
    # 1. FineWeb-Edu
    output_fineweb = "data/raw/fineweb_edu.txt"
    if not os.path.exists(output_fineweb):
        download_and_save(
            repo_id="HuggingFaceFW/fineweb-edu",
            config_name="sample-10BT",
            output_file=output_fineweb,
            num_samples=50000
        )
    else:
        print(f"⏩ FineWeb-Edu ya existe en {output_fineweb}. Saltando...")

    # 2. Cosmopedia v2
    output_cosmo = "data/raw/cosmopedia.txt"
    if not os.path.exists(output_cosmo):
        download_and_save(
            repo_id="HuggingFaceTB/cosmopedia-v2",
            config_name="cosmopedia-v2", 
            output_file=output_cosmo,
            num_samples=50000
        )
    else:
        print(f"⏩ Cosmopedia ya existe en {output_cosmo}. Saltando...")

    # 3. TinyStories (Versión limpia con tags)
    download_and_save(
        repo_id="roneneldan/TinyStories",
        config_name=None,
        output_file="data/raw/tinystories_v2.txt",
        num_samples=50000
    )

if __name__ == "__main__":
    main()
