from dataclasses import dataclass


@dataclass
class ModelConfig:
    ReplayBufferCapacity: int = 1000000 # 1mn from the paper
    NumActions: int = 4 # Breakout has 4 actions
    
    
    
    