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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from configs import ModelConfig
import cv2
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    def __init__(self, device=None):
        self.capacity = ModelConfig.ReplayBufferCapacity
        self.buffer = []
        self.position = 0
        self.device = device if device is not None else torch.device("cpu")  # Store on CPU by default

    def push(self, state, action, reward, next_state, done):
        # Convert tensors to CPU numpy arrays for storage
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        # Convert to torch tensors and move to device
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_shape, device=None):
        super(DQN, self).__init__()
        self.input_shape = input_shape  # (4,84,84)
        self.num_actions = ModelConfig.NumActions
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(self.device)
        self.fc = nn.Linear(self.feature_size(), 512).to(self.device)
        self.out = nn.Linear(512, self.num_actions).to(self.device)
        
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.out(x)

    def feature_size(self):
        # Create dummy tensor on the same device as the model
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape, device=self.device)))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            # Ensure state is a tensor on the correct device
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(np.float32(state))
            
            # Add batch dimension if needed and move to device
            if state.dim() == 3:
                state = state.unsqueeze(0)
                
            state = state.to(self.device)
            
            with torch.no_grad():
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action

class PreProcessing():
    @staticmethod
    def preprecess(image, device=None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        gray = gray/255.0
        tensor = torch.FloatTensor(gray).unsqueeze(0).squeeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

class DQNAgent:
    def __init__(self, input_shape, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models and explicitly move them to the specified device
        self.policy_net = DQN(input_shape)
        self.target_net = DQN(input_shape)
        
        # Explicitly move to the specified device
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        
        # After moving to device, load state dict from policy to target
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Create optimizer AFTER moving model to device
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayBuffer(device=self.device)

    def optimize_model(self, batch_size, gamma=0.99):
        if len(self.memory) < batch_size:
            return 0.0
            
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        
       
        # Reshape action tensor
        action = action.unsqueeze(1)
        
        # Reward clipping
        reward = torch.clamp(reward, -1, 1)
        
        # Compute Q(s_t, a) for the actions which were taken
        state_action_values = self.policy_net(state).gather(1, action)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = self.target_net(next_state).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = reward + (gamma * next_state_values * (1 - done))
        expected_state_action_values = torch.clamp(expected_state_action_values, -1, 1)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()

class FrameStacker:
    def __init__(self, num_frames=4, device=None):
        self.num_frames = num_frames
        self.frames = []
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def push(self, frame):
        # Ensure frame is on the correct device
        if isinstance(frame, torch.Tensor) and frame.device != self.device:
            frame = frame.to(self.device)
            
        if len(self.frames) >= self.num_frames:
            self.frames.pop(0)
        self.frames.append(frame)
        
    def get_state(self):
        # Duplicate the last frame to make it 4 frames
        while len(self.frames) < self.num_frames:
            self.frames.append(self.frames[-1])

        # Stack frames using torch
        return torch.stack(self.frames, dim=0)

def evaluate(agent, num_episodes=10, render=False):
    """
    Evaluate the agent's performance and track Q-value predictions
    """
    env = gym.make('ALE/Breakout-v5', render_mode="human" if render else None)
    frame_stacker = FrameStacker(device=agent.device)
    
    episode_rewards = []
    q_values = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        frame = PreProcessing.preprecess(obs, device=agent.device)
        frame_stacker.push(frame)
        
        episode_reward = 0
        episode_q_values = []
        done = False
        
        while not done:
            state = frame_stacker.get_state()
            
            # Get Q-values for all actions
            with torch.no_grad():
                # State is already on device
                q_value = agent.policy_net(state.unsqueeze(0))
                episode_q_values.append(q_value.max().item())  # Store maximum Q-value
                action = q_value.max(1)[1].item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            frame = PreProcessing.preprecess(obs, device=agent.device)
            frame_stacker.push(frame)
        
        episode_rewards.append(episode_reward)
        q_values.append(torch.tensor(episode_q_values).mean().item())
        
        print(f"Evaluation Episode {episode + 1}")
        print(f"  Reward: {episode_reward}")
        print(f"  Average Q-Value: {torch.tensor(episode_q_values).mean().item():.2f}")
        print(f"  Max Q-Value: {torch.tensor(episode_q_values).max().item():.2f}")
    
    env.close()
    
    return {
        'mean_reward': torch.tensor(episode_rewards).mean().item(),
        'std_reward': torch.tensor(episode_rewards).std().item(),
        'mean_q_value': torch.tensor(q_values).mean().item(),
        'std_q_value': torch.tensor(q_values).std().item()
    }

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






