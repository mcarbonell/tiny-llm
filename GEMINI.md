# 📜 Estándares y Metodología - Proyecto TinyThinker

Este documento define la normativa técnica y operativa del proyecto. Su cumplimiento es obligatorio para garantizar la integridad de los experimentos y la comparabilidad de los resultados.

## 1. Ciclo de Vida de Desarrollo
Toda nueva tarea o fase debe seguir estrictamente este flujo:
1.  **Implementación:** Escritura del código o scripts.
2.  **Pruebas Técnicas:** Ejecución de tests y validación manual.
3.  **Iteración:** Corrección de bugs y refinamiento hasta alcanzar estabilidad total.
4.  **Documentación:** Actualización de `PROJECT_STATUS.md`, `README.md` y logs.
5.  **Commit:** Consolidación en Git con mensajes descriptivos.
6.  **Siguiente Fase:** Solo tras completar los puntos anteriores.

## 2. Gestión de Activos y Seguridad
*   **Copias de Seguridad Obligatorias:** Antes de lanzar cualquier proceso que reescriba archivos (entrenamientos, limpiezas de dataset, etc.), se debe realizar una copia del original con un nombre semántico en una carpeta de respaldo (ej. `checkpoints/old/`). **Nunca se debe destruir un activo sin posibilidad de restauración.**
*   **Nombrado Semántico:** Los nombres de archivos, checkpoints y datasets deben ser descriptivos. 
    *   *Mal:* `ckpt_best.pt`, `data_v2.json`.
    *   *Bien:* `ckpt_base_corpus305M_v2.pt`, `dataset_toolcalling_cleaned_v1.json`.

## 3. Estándares de Logging
*   **Timestamp Relativo:** Las líneas de log deben usar el formato `[HH:MM:SS]` indicando el tiempo transcurrido desde el inicio.
*   **Header de Metadatos:** Toda ejecución debe imprimir un bloque con el hardware usado, hilos de CPU y todos los hiperparámetros.
*   **Sin Emojis:** Prohibidos para garantizar compatibilidad con terminales Windows.
*   **Trazabilidad de Ficheros Base:** En cada entrenamiento o finetune, se debe registrar:
    *   Nombre del fichero base utilizado (ej. `finetune_base: corpus_llamaware_v3.json`)
    *   Checksum/hash del fichero base (SHA256 o MD5) para evitar colisiones de nombres.
*   **Trazabilidad de Guardados:** Cada vez que se guarda un activo (checkpoint, modelo mejorado, dataset procesado), se debe registrar la ruta completa del fichero guardado (ej. `saved: checkpoints/ckpt_305M_v2.pt`).

## 4. Restricciones de Ejecución
*   **Sin Entrenamientos/Finetunes Automáticos:** El agente NO debe lanzar entrenamientos ni finetunes por cuenta propia. Debe proporcionar el comando al usuario para que lo ejecute manualmente.
*   **Sin Scripts con API Keys:** El agente NO debe ejecutar scripts que requieran claves de API (ej. `scripts/generate_rich_logic_curriculum.py` para generación de datasets sintéticos). Esto evita bloqueos por rate limits.
*   **Scripts Permitidos:** Solo scripts locales, de ejecución rápida, que no impliquen llamadas a APIs externas.
*   **Verificación de Ejecución Activa:** Antes de sugerir cualquier comando de entrenamiento o generación, el agente debe verificar si ya hay procesos en curso para evitar conflictos.

## 5. Unidades de Medida
*   **Tiempos:** Segundos (`s`) con dos decimales.
*   **Pérdida:** Cuatro decimales.
*   **Learning Rate:** Notación científica (`1.00e-04`).
