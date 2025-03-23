import gymnasium as gym
import ale_py
from DQN import DQNAgent, FrameStacker, PreProcessing, MetricLogger, evaluate
import torch
from utils import plot_training_metrics, plot_evaluation_metrics
from DQN import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv
gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import gymnasium as gym
import ale_py
from DQN import DQNAgent, FrameStacker, PreProcessing, MetricLogger, evaluate
import torch
from utils import plot_training_metrics, plot_evaluation_metrics
from DQN import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv

# Register ALE environments
gym.register_envs(ale_py)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import gymnasium as gym
import ale_py
from DQN import DQNAgent, FrameStacker, PreProcessing, MetricLogger, evaluate
import torch
from utils import plot_training_metrics, plot_evaluation_metrics
from DQN import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv

# Register ALE environments
gym.register_envs(ale_py)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(episodes, batch_size=32, eval_frequency=100):
    # Initialize environment with wrappers
    env = gym.make('ALE/Breakout-v5')
    env = NoopResetEnv(env)        # Perform random no-op actions at reset
    env = MaxAndSkipEnv(env)       # Skip frames and take max over consecutive frames
    env = EpisodicLifeEnv(env)     # Treat life loss as episode termination
    env = FireResetEnv(env)        # Press fire to start after reset
    
    # Initialize agent and utilities
    agent = DQNAgent(input_shape=(4, 84, 84), device=device)
    frame_stacker = FrameStacker(device=device)
    logger = MetricLogger()
    eval_history = []  # To store evaluation metrics over time
    
    # Global step counter and epsilon decay parameters
    global_step = 0
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay_steps = 100000
    decay_rate = (epsilon_start - epsilon_min) / epsilon_decay_steps
    
    # Constants
    INITIAL_MEMORY = 50000
    print(f"Filling replay buffer with {INITIAL_MEMORY} random experiences...")
    
    # Fill replay buffer with initial random experiences
    obs, _ = env.reset()
    frame = PreProcessing.preprecess(obs, device=device)
    frame_stacker.reset(frame)  # Reset frame stacker with initial frame
    
    while len(agent.memory) < INITIAL_MEMORY:
        state = frame_stacker.get_state()
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_frame = PreProcessing.preprecess(obs, device=device)
        frame_stacker.push(next_frame)
        next_state = frame_stacker.get_state()
        agent.memory.push(state, action, reward, next_state, done)
        global_step += 1
        if done:
            obs, _ = env.reset()
            frame = PreProcessing.preprecess(obs, device=device)
            frame_stacker.reset(frame)
    
    print("Memory filling complete. Starting training...")
    
    # Main training loop
    for episode in range(episodes):
        obs, _ = env.reset()
        frame = PreProcessing.preprecess(obs, device=device)
        frame_stacker.reset(frame)  # Reset frame stacker for new episode
        
        total_reward = 0
        done = False
        episode_q_values = []
        
        while not done:
            state = frame_stacker.get_state()
            # Calculate epsilon based on global_step
            epsilon = max(epsilon_min, epsilon_start - (decay_rate * global_step))
            action = agent.policy_net.act(state, epsilon)
            
            # Frame skipping: take the same action for 4 frames
            reward = 0
            for _ in range(4):
                obs, r, terminated, truncated, _ = env.step(action)
                reward += r
                if terminated or truncated:
                    done = True
                    break
            
            next_frame = PreProcessing.preprecess(obs, device=device)
            frame_stacker.push(next_frame)
            next_state = frame_stacker.get_state()
            
            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Optimize the model if enough samples are available
            if len(agent.memory) >= batch_size:
                loss = agent.optimize_model(batch_size)
            else:
                loss = 0.0  # Default loss value when not optimizing
            
            total_reward += reward
            global_step += 1
            
            # Update target network every 10,000 steps
            if global_step % 10000 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            # Collect Q-values for logging
            with torch.no_grad():
                q_value = agent.policy_net(state.unsqueeze(0)).max().item()
                episode_q_values.append(q_value)
        
        # Calculate mean Q-value for the episode
        mean_q = sum(episode_q_values) / len(episode_q_values) if episode_q_values else 0.0
        
        # Log episode metrics using MetricLogger
        logger.log_episode(total_reward, mean_q, loss, epsilon)
        
        # Evaluate and plot periodically
        if episode % eval_frequency == 0:
            # Perform evaluation
            metrics = evaluate(agent, num_episodes=5)
            print(f"Evaluation after episode {episode}: Mean Reward = {metrics['mean_reward']:.2f}")
            
            # Store evaluation results
            eval_history.append({
                'episode': episode,
                'mean_reward': metrics['mean_reward'],
                'std_reward': metrics['std_reward'],
                'mean_q_value': metrics['mean_q_value'],
                'std_q_value': metrics['std_q_value']
            })
            
            # Plot training metrics
            plot_training_metrics(logger)
            
            # Plot evaluation metrics if there is enough history
            if len(eval_history) > 1:
                plot_evaluation_metrics(eval_history)
        
        # Save checkpoint every 500 episodes
        if episode % 500 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': epsilon
            }, f'checkpoints/checkpoint_episode_{episode}.pt')
        
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f} , global_step: {global_step}"),

if __name__ == "__main__":
    # Run training with specified parameters
    train(episodes=20000, batch_size=32, eval_frequency=200)