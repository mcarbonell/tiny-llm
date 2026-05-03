import os
import glob
import re
import argparse
import matplotlib.pyplot as plt

def find_latest_log():
    log_files = glob.glob("logs/train_*.log")
    if not log_files:
        return None
    # Ordenar por fecha de modificación (el más reciente primero)
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[0]

def parse_log(log_path):
    iters = []
    losses = []
    
    val_iters = []
    val_train_losses = []
    val_losses = []
    
    # Expresiones regulares para capturar las métricas
    step_pattern = re.compile(r"iter\s+(\d+)\s+\|\s+loss\s+([0-9.]+)")
    val_pattern = re.compile(r"Iter\s+(\d+):\s+train_loss\s+([0-9.]+),\s+val_loss\s+([0-9.]+)")
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Buscar métricas de cada step
                step_match = step_pattern.search(line)
                if step_match:
                    iters.append(int(step_match.group(1)))
                    losses.append(float(step_match.group(2)))
                    continue
                    
                # Buscar métricas de validación
                val_match = val_pattern.search(line)
                if val_match:
                    val_iters.append(int(val_match.group(1)))
                    val_train_losses.append(float(val_match.group(2)))
                    val_losses.append(float(val_match.group(3)))
    except Exception as e:
        print(f"❌ Error al leer el archivo {log_path}: {e}")
                
    return iters, losses, val_iters, val_train_losses, val_losses

def plot_log(log_path):
    iters, losses, val_iters, val_train_losses, val_losses = parse_log(log_path)
    
    if not iters and not val_iters:
        print(f"⚠️ No se encontraron datos de pérdida válidos en {log_path}")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Dibujar la línea de loss por cada step (suele tener ruido, la hacemos más tenue)
    if iters:
        plt.plot(iters, losses, label='Step Loss', color='lightblue', alpha=0.6, linewidth=1.5)
        
    # Dibujar las evaluaciones periódicas (puntos y línea gruesa)
    if val_iters:
        plt.plot(val_iters, val_train_losses, label='Train Loss (Avg)', marker='o', color='blue', linestyle='dashed', linewidth=2)
        plt.plot(val_iters, val_losses, label='Validation Loss', marker='s', color='red', linestyle='solid', linewidth=2)
        
    plt.title(f'Curva de Entrenamiento de TinyThinker\nArchivo: {os.path.basename(log_path)}', fontsize=14, fontweight='bold')
    plt.xlabel('Iteraciones', fontsize=12)
    plt.ylabel('Pérdida (Cross-Entropy Loss)', fontsize=12)
    
    # Escala logarítmica si los valores iniciales son muy altos (>100) para no aplastar la gráfica
    if losses and max(losses) > 20:
        plt.yscale('log')
        plt.ylabel('Pérdida (Escala Logarítmica)')
        
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Guardar en el mismo directorio que el log, pero con extensión .png
    out_path = log_path.replace('.log', '.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica generada exitosamente en: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera gráficas de entrenamiento a partir de los logs.")
    parser.add_argument('--log', type=str, default=None, help='Ruta al archivo .log (opcional)')
    args = parser.parse_args()
    
    log_file = args.log
    
    # Auto-detección si no se pasa archivo
    if log_file is None:
        print("🔍 No se especificó log. Buscando el más reciente...")
        log_file = find_latest_log()
        if log_file:
            print(f"👉 Seleccionado: {log_file}")
        else:
            print("❌ No se encontraron archivos de log en el directorio 'logs/'.")
            exit(1)
    elif not os.path.exists(log_file):
        print(f"❌ El archivo especificado no existe: {log_file}")
        exit(1)
            
    plot_log(log_file)
