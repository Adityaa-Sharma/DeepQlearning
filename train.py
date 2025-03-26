# train.py
import gymnasium as gym
from DQN import *
import torch
import numpy as np
import ale_py

gym.register_envs(ale_py)

def create_env():
    # Register the environment with proper specification
    env = gym.make(
        "ALE/Breakout-v5",
        frameskip=1,  # Important: Disable built-in frame skipping
        render_mode="rgb_array",
        full_action_space=False
    )
    
    # Apply standard Atari preprocessing
    env=FireResetEnv(env)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,        # Combine 4 frames
        screen_size=84,      # Resize to 84x84
        terminal_on_life_loss=False,
        grayscale_obs=True  # Convert to grayscale
    )
    
    # Stack 4 frames as channels
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    return env

def train():
    env = create_env()
    agent = DQNAgent(input_shape=(4, 84, 84), num_actions=env.action_space.n)
    
    # Training parameters
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 1e6
    total_steps = 0
    update_target = 10000
    
    # Initial buffer filling
    print("Filling replay buffer...")
    state, _ = env.reset()
    state = torch.tensor(state.__array__(), device=device).float() / 255.0
    
    while len(agent.memory) < agent.batch_size:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(obs.__array__(), device=device).float() / 255.0
        done = terminated or truncated
        
        agent.memory.push(
            state.cpu(),
            action,
            np.clip(reward, -1, 1),
            next_state.cpu(),
            done
        )
        
        state = next_state if not done else torch.tensor(env.reset()[0].__array__(), 
                                             device=device).float() / 255.0
    
    # Main training loop
    print("Starting training...")
    episode = 0
    while True:
        state, _ = env.reset()
        state = torch.tensor(state.__array__(), device=device).float() / 255.0
        total_reward = 0
        done = False
        
        while not done:
            total_steps += 1
            
            # Epsilon decay
            epsilon = epsilon_min + (epsilon_start - epsilon_min) * \
                     torch.exp(torch.tensor(-total_steps/epsilon_decay)).item()
            
            # Select action
            action = agent.act(state.unsqueeze(0), epsilon)
            
            # Environment step
            obs, reward, terminated, truncated, _ = env.step(action.item())
            next_state = torch.tensor(obs.__array__(), device=device).float() / 255.0
            done = terminated or truncated
            
            # Store transition
            agent.memory.push(
                state.cpu(),
                action.item(),
                np.clip(reward, -1, 1),
                next_state.cpu(),
                done
            )
            
            # Optimize
            agent.optimize()
            
            # Update target network
            if total_steps % update_target == 0:
                agent.update_target()
            
            state = next_state
            total_reward += reward
            
        # Logging
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")
        episode += 1

if __name__ == "__main__":
    train()