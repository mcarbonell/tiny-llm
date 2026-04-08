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

## 4. Unidades de Medida
*   **Tiempos:** Segundos (`s`) con dos decimales.
*   **Pérdida:** Cuatro decimales.
*   **Learning Rate:** Notación científica (`1.00e-04`).
