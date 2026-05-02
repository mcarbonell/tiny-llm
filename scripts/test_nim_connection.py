import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('NVIDIA_API_KEY')
if not api_key:
    raise ValueError("NVIDIA_API_KEY not found in .env")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)

print("Probando conexión con NVIDIA NIM (deepseek-v4-pro)...")

try:
    completion = client.chat.completions.create(
      model="minimaxai/minimax-m2.7",
      messages=[{"role":"user","content":"Hello, reply with 'OK'"}],
      temperature=1,
      max_tokens=10,
      stream=True
    )

    print("Respuesta: ", end="", flush=True)
    for chunk in completion:
      if not getattr(chunk, "choices", None):
        continue
      if chunk.choices and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n✅ Conexión exitosa.")
except Exception as e:
    print(f"\n❌ Error: {e}")
