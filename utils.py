import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import imageio

def plot_training_metrics(logger, save_dir='plots'):
    """
    Plot and save training metrics
    
    Args:
        logger: MetricLogger instance with training history
        save_dir: Directory to save plot images
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Get episode numbers
    episodes = np.arange(1, len(logger.rewards) + 1)
    
    # Plot rewards
    axs[0].plot(episodes, logger.rewards, 'b-', alpha=0.3, label='Episode Reward')
    
    # Calculate and plot smoothed rewards
    if len(logger.rewards) >= 100:
        smoothed_rewards = []
        for i in range(len(logger.rewards) - 99):
            smoothed_rewards.append(np.mean(logger.rewards[i:i+100]))
        axs[0].plot(episodes[99:], smoothed_rewards, 'r-', label='100-Episode Moving Avg')
    
    axs[0].set_ylabel('Reward')
    axs[0].set_title('Training Rewards')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot Q-values
    axs[1].plot(episodes, logger.q_values, 'g-', alpha=0.3, label='Average Q-Value')
    
    # Calculate and plot smoothed Q-values
    if len(logger.q_values) >= 100:
        smoothed_q_values = []
        for i in range(len(logger.q_values) - 99):
            smoothed_q_values.append(np.mean(logger.q_values[i:i+100]))
        axs[1].plot(episodes[99:], smoothed_q_values, 'm-', label='100-Episode Moving Avg')
    
    axs[1].set_ylabel('Q-Value')
    axs[1].set_title('Average Q-Values')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot epsilon and loss together
    ax3 = axs[2]
    ax3.plot(episodes, logger.epsilons, 'y-', label='Epsilon')
    ax3.set_ylabel('Epsilon', color='y')
    ax3.set_xlabel('Episode')
    ax3.tick_params(axis='y', labelcolor='y')
    ax3.set_title('Exploration Rate and Loss')
    ax3.grid(True, alpha=0.3)
    
    # Create second y-axis for loss
    ax4 = ax3.twinx()
    ax4.plot(episodes, logger.losses, 'c-', alpha=0.5, label='Loss')
    ax4.set_ylabel('Loss', color='c')
    ax4.tick_params(axis='y', labelcolor='c')
    
    # Add legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_metrics.png", dpi=300)
    plt.close()

def plot_evaluation_metrics(eval_history, save_dir='plots'):
    """
    Plot evaluation metrics over time
    
    Args:
        eval_history: List of evaluation result dictionaries
        save_dir: Directory to save plot images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics
    steps = [entry['steps'] for entry in eval_history]
    rewards = [entry['mean_reward'] for entry in eval_history]
    reward_stds = [entry['std_reward'] for entry in eval_history]
    q_values = [entry['mean_q_value'] for entry in eval_history]
    q_value_stds = [entry['std_q_value'] for entry in eval_history]
    
    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot rewards with error bars
    axs[0].errorbar(steps, rewards, yerr=reward_stds, fmt='o-', capsize=5, label='Mean Reward')
    axs[0].set_ylabel('Reward')
    axs[0].set_title('Evaluation Rewards vs. Training Steps')
    axs[0].grid(True, alpha=0.3)
    
    # Plot Q-values with error bars
    axs[1].errorbar(steps, q_values, yerr=q_value_stds, fmt='o-', capsize=5, label='Mean Q-Value')
    axs[1].set_ylabel('Q-Value')
    axs[1].set_xlabel('Training Steps')
    axs[1].set_title('Evaluation Q-Values vs. Training Steps')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/evaluation_metrics.png", dpi=300)
    plt.close()

def save_agent_gif(agent, env, filepath='agent_play.gif'):
    """
    Records an episode of the agent playing and saves it as a GIF.

    Args:
        agent: The DQNAgent to evaluate.
        env: The Gymnasium environment.
        filepath: The path to save the GIF file.
    """
    frames = []
    agent.policy_net.eval()  # Set the network to evaluation mode
    
    device = next(agent.policy_net.parameters()).device

    state, _ = env.reset()
    frames.append(env.render())
    
    done = False
    while not done:
        state_tensor = torch.tensor(state.__array__(), dtype=torch.uint8)
        
        # Always act greedily (epsilon=0) for recording
        with torch.no_grad():
            action = agent.act(state_tensor.unsqueeze(0), 0.0)
        
        state, reward, terminated, truncated, _ = env.step(action.item())
        frames.append(env.render())
        done = terminated or truncated

    agent.policy_net.train()  # Set back to training mode

    # Save the collected frames as a GIF
    imageio.mimsave(filepath, frames, fps=30)
    print(f"Saved agent gameplay GIF to {filepath}")