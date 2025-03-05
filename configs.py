from dataclasses import dataclass


@dataclass
class ModelConfig:
    ReplayBuffer: int = 1000000 # 1mn from the paper
    
    
    