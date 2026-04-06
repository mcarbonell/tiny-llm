import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

@dataclass
class Config:
    # Model parameters
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 16384
    max_seq_len: int = 1024
    lora_r: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    use_gradient_checkpointing: bool = False  # Ahorra memoria a costa de re-cómputo en backward
    
    # Training parameters
    batch_size: int = 16
    seq_len: int = 256
    max_iters: int = 5000
    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    warmup_iters: int = 200
    eval_interval: int = 250
    eval_iters: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 4
    
    # Finetune parameters
    ft_batch_size: int = 4
    ft_max_iters: int = 500
    ft_learning_rate: float = 3e-5
    ft_eval_interval: int = 50
    ft_eval_iters: int = 10
    
    # Generation parameters
    temperature: float = 0.7
    top_k: int = 40
    max_new_tokens: int = 150
    
    # Paths
    data_path: str = "data/train_combined.bin"
    checkpoint_dir: str = "checkpoints"
    tokenizer_path: str = "model/tokenizer.json"
    
    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="TinyThinker Configuration")

        # YAML config base (opcional)
        parser.add_argument('--config', type=str, default=None,
                            help='Ruta a un archivo YAML de configuración. Los flags CLI sobreescriben los valores del YAML.')

        # Model
        parser.add_argument('--dim', type=int, default=None, help='Model dimension')
        parser.add_argument('--n_layers', type=int, default=None, help='Number of layers')
        parser.add_argument('--n_heads', type=int, default=None, help='Number of attention heads')
        parser.add_argument('--n_kv_heads', type=int, default=None, help='Number of KV heads (GQA)')
        parser.add_argument('--vocab_size', type=int, default=None, help='Vocabulary size')
        parser.add_argument('--max_seq_len', type=int, default=None, help='Maximum sequence length')
        parser.add_argument('--lora_r', type=int, default=None, help='LoRA rank (0 disables LoRA)')
        parser.add_argument('--lora_alpha', type=float, default=None, help='LoRA alpha scaling')
        parser.add_argument('--lora_dropout', type=float, default=None, help='LoRA dropout')
        parser.add_argument('--use_gradient_checkpointing', action='store_true', default=None,
                            help='Activa gradient checkpointing para reducir uso de memoria (a costa de velocidad).')

        # Training
        parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
        parser.add_argument('--seq_len', type=int, default=None, help='Sequence length')
        parser.add_argument('--max_iters', type=int, default=None, help='Maximum iterations')
        parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
        parser.add_argument('--min_lr', type=float, default=None, help='Minimum learning rate')
        parser.add_argument('--warmup_iters', type=int, default=None, help='Warmup iterations')
        parser.add_argument('--eval_interval', type=int, default=None, help='Evaluation interval')
        parser.add_argument('--eval_iters', type=int, default=None, help='Evaluation iterations')
        parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
        parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping')
        parser.add_argument('--grad_accum_steps', type=int, default=None, help='Gradient accumulation steps')

        # Finetune
        parser.add_argument('--ft_batch_size', type=int, default=None, help='Finetune batch size')
        parser.add_argument('--ft_max_iters', type=int, default=None, help='Finetune max iterations')
        parser.add_argument('--ft_learning_rate', type=float, default=None, help='Finetune learning rate')
        parser.add_argument('--ft_eval_interval', type=int, default=None, help='Finetune eval interval')
        parser.add_argument('--ft_eval_iters', type=int, default=None, help='Finetune eval iterations')

        # Generation
        parser.add_argument('--temperature', type=float, default=None, help='Generation temperature')
        parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
        parser.add_argument('--max_new_tokens', type=int, default=None, help='Max new tokens')

        # Paths
        parser.add_argument('--data_path', type=str, default=None, help='Data path')
        parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
        parser.add_argument('--tokenizer_path', type=str, default=None, help='Tokenizer path')

        args = parser.parse_args()

        # 1. Partir de los defaults del dataclass
        cfg = cls()

        # 2. Aplicar valores del YAML (si existe --config)
        if args.config is not None:
            if not _YAML_AVAILABLE:
                raise ImportError("--config requiere PyYAML. Instala con: pip install PyYAML")
            if not os.path.exists(args.config):
                raise FileNotFoundError(f"Archivo de config YAML no encontrado: {args.config}")
            with open(args.config, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f) or {}
            valid_fields = {f.name for f in cfg.__dataclass_fields__.values()}
            for key, value in yaml_data.items():
                if key in valid_fields:
                    setattr(cfg, key, value)
                else:
                    print(f"[config] Advertencia: clave YAML desconocida ignorada: '{key}'")

        # 3. Los flags CLI sobreescriben el YAML (solo si el usuario los pasó explicitamente)
        cli_dict = {k: v for k, v in vars(args).items() if k != 'config' and v is not None}
        for key, value in cli_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        return cfg

    def save_yaml(self, path: str):
        """Guarda la configuración actual en un archivo YAML."""
        if not _YAML_AVAILABLE:
            raise ImportError("save_yaml requiere PyYAML. Instala con: pip install PyYAML")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {k: v for k, v in self.__dict__.items()}
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        print(f"[config] Configuración guardada en: {path}")