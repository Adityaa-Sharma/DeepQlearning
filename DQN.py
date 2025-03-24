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
        self.policy_net = DQN(input_shape).to(self.device)
        self.target_net = DQN(input_shape).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.00025)  # Update this line
        self.memory = ReplayBuffer(device=self.device)

    def optimize_model(self, batch_size, gamma=0.99):  # Remove step_count parameter
        if len(self.memory) < batch_size:
            return 0.0
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        action = action.unsqueeze(1)
        reward = torch.clamp(reward, -1, 1)
        
        state_action_values = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            next_state_values = self.target_net(next_state).max(1)[0]
            expected_state_action_values = reward + (gamma * next_state_values * (1 - done))
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
from collections import deque
import torch

class FrameStacker:
    
    def __init__(self, num_frames=4, device=None):
       
        self.num_frames = num_frames
        self.device = device if device is not None else torch.device("cpu")
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, initial_frame):
        
        self.frames = deque([initial_frame] * self.num_frames, maxlen=self.num_frames)
    
    def push(self, frame):
        
        self.frames.append(frame)
    
    def get_state(self):
        
        return torch.stack(list(self.frames), dim=0)

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
        frame_stacker.reset(frame)  # This duplicates the frame 4 times
        
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

# Add at the beginning of DQN.py or create new wrappers.py
import numpy as np
from collections import deque

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset."""
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over."""
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted."""
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                done = True
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)






