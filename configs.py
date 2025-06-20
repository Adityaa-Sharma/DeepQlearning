from dataclasses import dataclass


@dataclass
class ModelConfig:
    ReplayBufferCapacity: int = 1000000 
    NumActions: int = 4 
    lr: float = 0.0000625 
    gamma: float = 0.99
    target_update: int = 10000
    batch_size: int = 32
    epsilon_start: float = 1.0  # Should be float
    epsilon_min: float = 0.1    # Should be float
    epsilon_decay: int = 250000 # Paper: 1M frames. With frame_skip=4, this is 1,000,000 / 4 = 250,000 agent steps.
    total_steps: int = 0
    update_target: int = 10000
    eval_freq: int = 50000  # Evaluate every 50,000 agent steps
    eval_episodes: int = 30  # Number of episodes for evaluation: 30



