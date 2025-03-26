#modules
#1) Replay bufffer
#2) DQN
#3) DQN with target network
#4) sampleing from replay buffer
#5) preprocesing of the input to 84x84x4
#6) training the network
#7) testing the network-> reward and preicted Q Vlue estimation
#8) saving the model


# choosen game : Breakout


#dqn
# DQN.py
# DQN.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T
from collections import deque
import random
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity=int(1e5)):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state.cpu().clone(),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            next_state.cpu().clone(),
            torch.tensor(done, dtype=torch.float32)
        ))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        return (
            torch.stack(states).to(device),
            torch.stack(actions).unsqueeze(1).to(device),  # Add dimension for gathering
            torch.stack(rewards).to(device),
            torch.stack(next_states).to(device),
            torch.stack(dones).to(device)
        )

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._conv_out_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    def _conv_out_size(self, shape):
        return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)
        
    def forward(self, x):
        x = x.float().div_(255)
        return self.fc(self.conv(x).view(x.size(0), -1))

class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.policy_net = DQN(input_shape, num_actions).to(device)
        self.target_net = DQN(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), 
                                     lr=0.00025, alpha=0.95, eps=0.01)
        self.memory = ReplayBuffer()
        self.batch_size = 32
        self.gamma = 0.99
        
    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        return torch.tensor([[random.randint(0, self.policy_net.fc[-1].out_features-1)]], 
                           device=device, dtype=torch.long)
    
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)  # Shape: [batch_size, 1]
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]  # Shape: [batch_size]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            target_q = target_q.unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
# Environment Wrappers
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        
    def reset(self):
        self.env.reset()
        noops = random.randint(1, self.noop_max)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done: self.env.reset()
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self):
        self.env.reset()
        obs, _, done, _, _ = self.env.step(1)  # Send FIRE action (action=1)
        if done:
            self.env.reset()
        return obs
    

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done: break
        return obs, total_reward, done, info
class MetricLogger:
    """
    Logger to track training metrics
    """
    def __init__(self):
        self.rewards = []
        self.q_values = []
        self.losses = []
        self.epsilons = []
    
    def log_episode(self, reward, q_value, loss, epsilon):
        self.rewards.append(float(reward))
        self.q_values.append(float(q_value))
        self.losses.append(float(loss))
        self.epsilons.append(float(epsilon))
    
    def get_metrics(self):
        if len(self.rewards) > 0:
            return {
                'last_100_mean_reward': torch.tensor(self.rewards[-100:]).mean().item(),
                'last_100_mean_q': torch.tensor(self.q_values[-100:]).mean().item(),
                'last_100_mean_loss': torch.tensor(self.losses[-100:]).mean().item(),
                'current_epsilon': self.epsilons[-1]
            }
        else:
            return {
                'last_100_mean_reward': 0.0,
                'last_100_mean_q': 0.0,
                'last_100_mean_loss': 0.0,
                'current_epsilon': None
            }
    
    def save_data(self, folder='tensors'):
        """
        Save all logged metrics as tensors
        """
        import os
        os.makedirs(folder, exist_ok=True)
        
        torch.save(torch.tensor(self.rewards), f"{folder}/rewards.pt")
        torch.save(torch.tensor(self.q_values), f"{folder}/q_values.pt")
        torch.save(torch.tensor(self.losses), f"{folder}/losses.pt")
        torch.save(torch.tensor(self.epsilons), f"{folder}/epsilons.pt")
    
    @classmethod
    def load_data(cls, folder='tensors'):
        """
        Load metrics from saved tensors
        """
        logger = cls()
        try:
            logger.rewards = torch.load(f"{folder}/rewards.pt").tolist()
            logger.q_values = torch.load(f"{folder}/q_values.pt").tolist()
            logger.losses = torch.load(f"{folder}/losses.pt").tolist()
            logger.epsilons = torch.load(f"{folder}/epsilons.pt").tolist()
        except FileNotFoundError:
            print("Warning: Could not load all tensor files.")
        return logger
