import argparse

def calculate_compute(n_params, n_tokens, tflops, efficiency, cost_per_hour=0.0):
    """
    Calcula el tiempo y coste de entrenamiento basado en la fórmula C = 6 * N * T
    
    n_params: Número de parámetros (ej: 100e6 para 100M)
    n_tokens: Número de tokens en el dataset (ej: 1e9 para 1B)
    tflops: TFLOPS teóricos del hardware en FP16/BF16
    efficiency: Factor de utilización real (0.0 a 1.0, típico 0.3-0.5)
    cost_per_hour: Precio en USD por hora de la GPU
    """
    
    # Total FLOPs (6 * N * T)
    total_flops = 6 * n_params * n_tokens
    
    # TFLOPS efectivos
    effective_tflops = tflops * efficiency
    effective_flops_per_sec = effective_tflops * 1e12
    
    # Tiempo en segundos
    total_seconds = total_flops / effective_flops_per_sec
    
    # Resultados
    hours = total_seconds / 3600
    days = hours / 24
    tokens_per_sec = n_tokens / total_seconds
    total_cost = hours * cost_per_hour
    
    return {
        "total_flops": total_flops,
        "total_seconds": total_seconds,
        "hours": hours,
        "days": days,
        "tokens_per_sec": tokens_per_sec,
        "total_cost": total_cost,
        "effective_tflops": effective_tflops
    }

def main():
    parser = argparse.ArgumentParser(description="Calculadora de Cómputo para LLMs (TinyThinker)")
    parser.add_argument("--params", type=float, default=100e6, help="Parámetros del modelo (ej: 100e6)")
    parser.add_argument("--tokens", type=float, default=1e9, help="Tokens del dataset (ej: 1e9 para 1B)")
    
    # Perfiles de Hardware
    hw_profiles = {
        "780m": {"tflops": 16.5, "eff": 0.35, "cost": 0.0, "name": "Radeon 780M (Local iGPU)"},
        "4090": {"tflops": 330.0, "eff": 0.45, "cost": 0.35, "name": "NVIDIA RTX 4090 (RunPod/Vast)"},
        "a100": {"tflops": 312.0, "eff": 0.55, "cost": 2.5, "name": "NVIDIA A100 80GB (Modal/Cloud)"},
        "h100": {"tflops": 989.0, "eff": 0.40, "cost": 4.0, "name": "NVIDIA H100 (Cloud)"}
    }
    
    parser.add_argument("--hw", choices=hw_profiles.keys(), default="780m", help="Perfil de hardware a usar")
    args = parser.parse_args()
    
    hw = hw_profiles[args.hw]
    res = calculate_compute(args.params, args.tokens, hw["tflops"], hw["eff"], hw["cost"])
    
    print(f"\n" + "="*50)
    print(f"📊 ESTIMACIÓN DE ENTRENAMIENTO: {hw['name']}")
    print(f"="*50)
    print(f"Configuración:")
    print(f"  - Parámetros: {args.params/1e6:.1f} M")
    print(f"  - Tokens:     {args.tokens/1e9:.2f} B")
    print(f"  - Hardware:   {hw['tflops']} TFLOPS (FP16/BF16)")
    print(f"  - Eficiencia: {hw['eff']*100:.1f}% (Utilización real)")
    
    print(f"\nResultados:")
    print(f"  - Cómputo Total: {res['total_flops']:.2e} FLOPs")
    print(f"  - Velocidad:     {res['tokens_per_sec']:.0f} tokens/seg")
    print(f"  - Tiempo:        {res['hours']:.2f} horas ({res['days']:.2f} días)")
    
    if res['total_cost'] > 0:
        print(f"  - Coste Est.:    ${res['total_cost']:.2f} USD")
    else:
        print(f"  - Coste Est.:    GRATIS (Hardware local)")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
