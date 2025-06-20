# train.py
import gymnasium as gym
from DQN import *
import torch
import numpy as np
import ale_py
from utils import plot_training_metrics, plot_evaluation_metrics
import os
import logging
from configs import ModelConfig

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'training.log')),
        logging.StreamHandler()  # Also output to console
    ]
)
logger = logging.getLogger(__name__)

gym.register_envs(ale_py)

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots', exist_ok=True)

metrics_logger = MetricLogger()

def evaluate_agent(agent, env, eval_episodes=ModelConfig.eval_episodes, fixed_states=None):
    """
    Runs a comprehensive evaluation of the agent's performance.
    """
    agent.policy_net.eval()  # Set the network to evaluation mode
    total_rewards = []
    
    for _ in range(eval_episodes):
        state, _ = env.reset()
        state = torch.tensor(state.__array__(), device=device).float() / 255.0
        done = False
        episode_reward = 0
        while not done:
            # Always act greedily (epsilon=0) in evaluation
            with torch.no_grad():
                action = agent.policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            next_state = torch.tensor(obs.__array__(), device=device).float() / 255.0
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    # Calculate average max Q on the fixed set of states
    avg_max_q = 0
    if fixed_states is not None:
        with torch.no_grad():
            q_values = agent.policy_net(fixed_states)
            avg_max_q = q_values.max(1)[0].mean().item()

    agent.policy_net.train()  # Set back to training mode
    return mean_reward, std_reward, avg_max_q

def create_env():
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="rgb_array",
        full_action_space=False,
        frameskip=1
        
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
    eval_env = create_env()  # A separate environment for evaluation
    agent = DQNAgent(input_shape=(4, 84, 84), num_actions=env.action_space.n)
    
    # Hyperparameters
    epsilon_start = ModelConfig.epsilon_start
    epsilon_min = ModelConfig.epsilon_min
    epsilon_decay = ModelConfig.epsilon_decay  # Faster decay
    total_steps = 0
    update_target = ModelConfig.update_target
    min_replay_size = ModelConfig.ReplayBufferCapacity // 10  # Start training after 10% of buffer is filled
    eval_freq = ModelConfig.eval_freq  # Evaluate every 50,000 steps
    next_eval_step = eval_freq
    
    # Create a fixed set of states for Q-value evaluation (as per the paper)
    logger.info("Creating fixed states for Q-value evaluation...")
    fixed_states_buffer = []
    state, _ = eval_env.reset()
    for _ in range(10000):  # Collect 10k states
        action = eval_env.action_space.sample()
        obs, _, terminated, truncated, _ = eval_env.step(action)
        state_tensor = torch.tensor(obs.__array__(), device=device).float() / 255.0
        fixed_states_buffer.append(state_tensor)
        if terminated or truncated:
            obs, _ = eval_env.reset()
            state_tensor = torch.tensor(obs.__array__(), device=device).float() / 255.0
    fixed_states = torch.stack(fixed_states_buffer).to(device)
    logger.info(f"Collected {len(fixed_states)} states for evaluation.")

    # Initialize replay buffer
    logger.info("Filling replay buffer...")
    state, _ = env.reset()
    state = torch.tensor(state.__array__(), device=device).float() / 255.0  # Single normalization
    
    while len(agent.memory) < min_replay_size:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(obs.__array__(), device=device).float() / 255.0
        done = terminated or truncated
        
        # reward clipping {-1,0,1}
        clipped_reward = np.sign(reward) if reward != 0 else 0
        agent.memory.push(
            state.cpu(),
            action,
            clipped_reward,  
            next_state.cpu(),
            done
        )
        state = next_state if not done else torch.tensor(env.reset()[0].__array__(), 
                                             device=device).float() / 255.0
    
    # Training loop
    logger.info("Starting training...")
    episode = 0
    eval_history = []
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
            
            # reward clipping {-1,0,1}
            clipped_reward = np.sign(reward) if reward != 0 else 0
            
            agent.memory.push(
                state.cpu(),
                action.item(),
                clipped_reward,  
                next_state.cpu(),
                done
            )
            
            loss = agent.optimize()
            if loss is not None:
                episode_losses.append(loss)
            
            if total_steps % update_target == 0:
                agent.update_target()
            
            # --- Periodic Evaluation ---
            if total_steps ==0 or (total_steps % eval_freq == 0):
                mean_reward, std_reward, avg_max_q = evaluate_agent(
                    agent, eval_env, eval_episodes=30, fixed_states=fixed_states
                )
                logger.info(f"\n--- Evaluation at {total_steps} steps ---")
                logger.info(f"    Avg Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                logger.info(f"    Avg Max Q: {avg_max_q:.2f}")
                logger.info(f"------------------------------------")
                
                eval_history.append({
                    'steps': total_steps,
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'mean_q_value': avg_max_q,
                    'std_q_value': 0 # Placeholder, can be calculated if needed
                })
                
                # Update the plot for evaluation metrics
                plot_evaluation_metrics(eval_history, save_dir='plots')

                next_eval_step += eval_freq

            state = next_state
            total_reward += reward
        
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        metrics_logger.log_episode(total_reward, avg_q_value, avg_loss, epsilon)
        metrics = metrics_logger.get_metrics()
        logger.info(f"Episode {episode}, Reward: {total_reward}, Eps: {epsilon:.3f}, "
              f"Avg Q: {avg_q_value:.2f}, Loss: {avg_loss:.4f}, Steps: {total_steps}, "
              f"Frames: {total_steps * 4}, Avg Reward (100): {metrics['last_100_mean_reward']:.2f}")
        
        if episode % 10 == 0 and episode > 0:
            # checkpoint_path = f"checkpoints/dqn_episode_{episode}.pt"
            # torch.save({
            #     'episode': episode,
            #     'policy_net_state_dict': agent.policy_net.state_dict(),
            #     'target_net_state_dict': agent.target_net.state_dict(),
            #     'optimizer_state_dict': agent.optimizer.state_dict(),
            #     'epsilon': epsilon,
            #     'total_steps': total_steps,
            # }, checkpoint_path)
            # metrics_logger.save_data(folder='tensors')
            plot_training_metrics(metrics_logger, save_dir='plots')
            # plot_evaluation_metrics(metrics_logger, save_dir='plots/evaluation')
        
        episode += 1

if __name__ == "__main__":
    train()