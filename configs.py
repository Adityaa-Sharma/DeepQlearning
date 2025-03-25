from dataclasses import dataclass


@dataclass
class ModelConfig:
    ReplayBufferCapacity: int = 1000000 
    NumActions: int = 4 
    lr: float = 0.0000625 
    gamma: float = 0.99
    target_update: int = 10000
    batch_size: int = 32
    epsilon_start: int = 1.0
    epsilon_min: int = 0.1
    epsilon_decay: int = 1e6
    total_steps: int = 0
    update_target: int = 10000
    
    
    
    