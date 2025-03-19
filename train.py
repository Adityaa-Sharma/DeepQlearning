import gymnasium as gym
import ale_py
from DQN import DQNAgent, FrameStacker, PreProcessing, MetricLogger, evaluate
import torch

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(episodes, batch_size=32, eval_frequency=100):
    env = gym.make('ALE/Breakout-v5')
    
    # Initialize agent with device
    print(f"Creating agent on {device}")
    agent = DQNAgent(input_shape=(4, 84, 84), device=device)
    
    # Verify device placement
    print(f"Policy net device: {next(agent.policy_net.parameters()).device}")
    print(f"Target net device: {next(agent.target_net.parameters()).device}")
    
    frame_stacker = FrameStacker(device=device)
    logger = MetricLogger()
    
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = (epsilon - epsilon_min) / 1000000
    
    for episode in range(episodes):
        obs, _ = env.reset()
        frame = PreProcessing.preprecess(obs, device=device)
        frame_stacker.push(frame)
        
        total_reward = 0
        loss = 0
        done = False
        
        # Log metrics
        episode_q_values = []
        while not done:
            state = frame_stacker.get_state()
            action = agent.policy_net.act(state, epsilon)
            
            # Frame skipping (k=4) but getting the reward
            reward = 0
            for _ in range(4):
                obs, r, terminated, truncated, _ = env.step(action)
                reward += r
                if terminated or truncated:
                    done = True
                    break
                    
            frame = PreProcessing.preprecess(obs, device=device)
            frame_stacker.push(frame)
            next_state = frame_stacker.get_state()
            
       
            agent.memory.push(state, action, reward, next_state, done)
            
            if len(agent.memory) >= batch_size:
                loss = agent.optimize_model(batch_size)
            
            # Update target network periodically
            if episode % 10 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                
            total_reward += reward
            epsilon = max(epsilon_min, epsilon - epsilon_decay)
            
            # Get Q-values
            with torch.no_grad():
                q_value = agent.policy_net(state.unsqueeze(0))
                episode_q_values.append(q_value.max().item())
        
        # Get loss value
        loss_value = agent.optimize_model(batch_size) if len(agent.memory) >= batch_size else 0.0
        
        # Calculate mean Q-value with PyTorch
        mean_q = torch.tensor(episode_q_values).mean().item() if episode_q_values else 0.0
        
        # Log metrics
        logger.log_episode(
            reward=float(total_reward),
            q_value=float(mean_q),
            loss=float(loss_value),
            epsilon=float(epsilon)
        )
        
        # Periodic evaluation
        if episode % eval_frequency == 0:
            metrics = evaluate(agent, num_episodes=5)
            print("\nEvaluation Results:")
            print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"Mean Q-Value: {metrics['mean_q_value']:.2f} ± {metrics['std_q_value']:.2f}\n")
            
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
        
if __name__ == "__main__":
    train(episodes=1000, batch_size=32, eval_frequency=100)