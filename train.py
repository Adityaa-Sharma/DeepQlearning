# train.py
import gymnasium as gym
from DQN import *
import torch
import numpy as np
import ale_py
from utils import plot_training_metrics, plot_evaluation_metrics
import os

gym.register_envs(ale_py)

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots', exist_ok=True)

logger = MetricLogger()

def create_env():
    env = gym.make(
        "ALE/Breakout-v5",
        frameskip=1,
        render_mode="rgb_array",
        full_action_space=False
    )
    env = FireResetEnv(env)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,  
        grayscale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def train():
    env = create_env()
    agent = DQNAgent(input_shape=(4, 84, 84), num_actions=env.action_space.n)
    
    # Hyperparameters
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 50_000  # Faster decay
    total_steps = 0
    update_target = 10000
    min_replay_size = 50000
    
    # Initialize replay buffer
    print("Filling replay buffer...")
    state, _ = env.reset()
    state = torch.tensor(state.__array__(), device=device).float() / 255.0  # Single normalization
    
    while len(agent.memory) < min_replay_size:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(obs.__array__(), device=device).float() / 255.0
        done = terminated or truncated
        
        agent.memory.push(
            state.cpu(),
            action,
            reward,  # No clipping
            next_state.cpu(),
            done
        )
        state = next_state if not done else torch.tensor(env.reset()[0].__array__(), 
                                             device=device).float() / 255.0
    
    # Training loop
    print("Starting training...")
    episode = 0
    while True:
        state, _ = env.reset()
        state = torch.tensor(state.__array__(), device=device).float() / 255.0
        total_reward = 0
        done = False
        episode_q_values = []
        episode_losses = []
        
        while not done:
            total_steps += 1
            epsilon = max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min) * (total_steps / epsilon_decay))
            
            action = agent.act(state.unsqueeze(0), epsilon)
            
            with torch.no_grad():
                q_value = agent.policy_net(state.unsqueeze(0)).max().item()
                episode_q_values.append(q_value)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            next_state = torch.tensor(obs.__array__(), device=device).float() / 255.0
            done = terminated or truncated
            
            agent.memory.push(
                state.cpu(),
                action.item(),
                reward,  # No clipping
                next_state.cpu(),
                done
            )
            
            loss = agent.optimize()
            if loss is not None:
                episode_losses.append(loss)
            
            if total_steps % update_target == 0:
                agent.update_target()
            
            state = next_state
            total_reward += reward
        
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        logger.log_episode(total_reward, avg_q_value, avg_loss, epsilon)
        metrics = logger.get_metrics()
        print(f"Episode {episode}, Reward: {total_reward}, Eps: {epsilon:.3f}, "
              f"Avg Q: {avg_q_value:.2f}, Loss: {avg_loss:.4f}, Steps: {total_steps}, "
              f"Avg Reward (100): {metrics['last_100_mean_reward']:.2f}")
        
        if episode % 1000 == 0 and episode > 0:
            checkpoint_path = f"checkpoints/dqn_episode_{episode}.pt"
            torch.save({
                'episode': episode,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': epsilon,
                'total_steps': total_steps,
            }, checkpoint_path)
            logger.save_data(folder='tensors')
            plot_training_metrics(logger, save_dir='plots')
        
        episode += 1

if __name__ == "__main__":
    train()