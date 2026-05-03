# 🚀 Guía de Migración a Cloud Compute (GPU)

Este documento contiene los pasos exactos para migrar el entrenamiento de TinyThinker a la nube, reduciendo el tiempo de iteración de ~21s (CPU local) a <1s (GPU).

---

## Opción A: Google Colab (Ideal para empezar rápido)

Google Colab proporciona un entorno Jupyter Notebook con GPUs gratuitas (T4) o de pago (L4/A100 con Colab Pro).

### Pasos:
1. Sube tu código actual a un repositorio de GitHub (puede ser privado). Asegúrate de ignorar las carpetas `data/` y `checkpoints/` si son muy pesadas (puedes volver a generar los datos en la nube).
2. Abre un nuevo Notebook en [Google Colab](https://colab.research.google.com/).
3. Ve a **Entorno de ejecución > Cambiar tipo de entorno de ejecución** y selecciona **GPU (T4, L4, o A100)**.
4. En una celda del notebook, ejecuta los siguientes comandos:

```python
# 1. Clonar el repositorio (si es privado, genera un Personal Access Token en GitHub)
# Reemplaza URL_DEL_REPO con tu URL real
!git clone URL_DEL_REPO tiny-thinker

# 2. Entrar al directorio
%cd tiny-thinker

# 3. Instalar dependencias
!pip install -r requirements.txt

# 4. (Opcional) Si no subiste los datos, regenerarlos
# !python scripts/prepare_data.py

# 5. Iniciar el entrenamiento
# PyTorch detectará CUDA automáticamente
!python scripts/train.py --config configs/benchmark_spectral_v7_hd.yaml
```

**Nota para Colab:** Para evitar que la sesión se desconecte por inactividad, asegúrate de dejar la pestaña abierta. Si el entrenamiento toma muchas horas, Colab puede reiniciarlo, así que es clave que el script guarde checkpoints frecuentemente.

---

## Opción B: RunPod / Lambda Labs (Control total por SSH)

Esta opción es ideal si prefieres usar VSCode remotamente o SSH, tal como trabajas en local, pagando por hora ($0.30 - $0.50/h).

### Pasos:
1. Crea una cuenta en [RunPod](https://www.runpod.io/) o [Lambda Labs](https://lambdalabs.com/).
2. Despliega una instancia (Pod) con una GPU (ej. RTX 3090, 4090, o A5000). Selecciona una plantilla base de PyTorch (suele llamarse `RunPod PyTorch` o similar).
3. Conéctate a tu Pod vía SSH usando la terminal local o VSCode Remote SSH.
4. Una vez dentro de la terminal de la GPU:

```bash
# 1. Clonar tu repositorio
git clone URL_DEL_REPO tiny-thinker
cd tiny-thinker

# 2. Instalar requerimientos (el entorno ya tendrá PyTorch con CUDA instalado)
pip install -r requirements.txt

# 3. (Opcional) Regenerar datos si no se subieron
# python scripts/prepare_data.py

# 4. Lanzar el entrenamiento usando nohup o tmux para que no se corte 
# si pierdes la conexión SSH
nohup python scripts/train.py --config configs/benchmark_spectral_v7_hd.yaml > entrenamiento.log 2>&1 &

# Puedes ver el progreso con:
tail -f entrenamiento.log
```

---

## 🛠️ Consejos Adicionales para la Nube

*   **Sincronización de Checkpoints:** Si usas RunPod/Lambda, tus datos persisten mientras pagues el almacenamiento. En Colab, los datos se borran al cerrar la sesión. Para Colab, añade un script o usa Google Drive para guardar los `.pt`:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    # Luego, configura tu train.py para que guarde los checkpoints en /content/drive/MyDrive/...
    ```
*   **Descarga de Datos:** Si el script `prepare_data.py` tarda mucho, sube `train_v2_32k.bin` a Google Drive o un bucket S3 y descárgalo con `wget` o `gdown` directamente en la instancia de la nube.
*   **Monitoreo:** El archivo `train.py` ya imprime los logs. Al usar GPUs, notarás que `time` por iteración baja a milisegundos.
