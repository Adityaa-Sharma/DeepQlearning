from dataclasses import dataclass


@dataclass
class ModelConfig:
    ReplayBufferCapacity: int = 1000000 # 1mn from the paper
    NumActions: int = 4 # Breakout has 4 actions
    lr: float = 0.0000625 
    gamma: float = 0.99
    target_update: int = 10000
    batch_size: int = 32
    
    
    
    