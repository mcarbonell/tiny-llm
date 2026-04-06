import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Model parameters
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 16384
    max_seq_len: int = 1024
    
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
    data_path: str = "data/train.bin"
    checkpoint_dir: str = "checkpoints"
    tokenizer_path: str = "model/tokenizer.json"
    
    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="TinyThinker Configuration")
        
        # Model
        parser.add_argument('--dim', type=int, default=256, help='Model dimension')
        parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
        parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
        parser.add_argument('--n_kv_heads', type=int, default=4, help='Number of KV heads (GQA)')
        parser.add_argument('--vocab_size', type=int, default=16384, help='Vocabulary size')
        parser.add_argument('--max_seq_len', type=int, default=1024, help='Maximum sequence length')
        
        # Training
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('--seq_len', type=int, default=256, help='Sequence length')
        parser.add_argument('--max_iters', type=int, default=5000, help='Maximum iterations')
        parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
        parser.add_argument('--warmup_iters', type=int, default=200, help='Warmup iterations')
        parser.add_argument('--eval_interval', type=int, default=250, help='Evaluation interval')
        parser.add_argument('--eval_iters', type=int, default=20, help='Evaluation iterations')
        parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
        parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
        parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
        
        # Finetune
        parser.add_argument('--ft_batch_size', type=int, default=4, help='Finetune batch size')
        parser.add_argument('--ft_max_iters', type=int, default=500, help='Finetune max iterations')
        parser.add_argument('--ft_learning_rate', type=float, default=3e-5, help='Finetune learning rate')
        parser.add_argument('--ft_eval_interval', type=int, default=50, help='Finetune eval interval')
        parser.add_argument('--ft_eval_iters', type=int, default=10, help='Finetune eval iterations')
        
        # Generation
        parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
        parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling')
        parser.add_argument('--max_new_tokens', type=int, default=150, help='Max new tokens')
        
        # Paths
        parser.add_argument('--data_path', type=str, default='data/train.bin', help='Data path')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
        parser.add_argument('--tokenizer_path', type=str, default='model/tokenizer.json', help='Tokenizer path')
        
        args = parser.parse_args()
        return cls(**vars(args))