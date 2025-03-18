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
import gym
from configs import ModelConfig
import cv2


class ReplayBuffer:
    def __init__(self):
        self.capacity = ModelConfig.ReplayBufferCapacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_shape):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = ModelConfig.NumActions
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(self.feature_size(), 512)
        self.out = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.out(x)

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1) # this will output 3136

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0) # adding the batch dimension
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action
    

class PreProcessing():
    # This will make the input as 210x160x3 to 84x84x4
    def preprecess(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        gray=gray/255.0
        return torch.FloatTensor(gray).unsqueeze(0)

class DQNAgent:
    def __init__(self, input_shape):
        self.policy_net = DQN(input_shape)
        self.target_net = DQN(input_shape)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # only update after 10000 steps
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayBuffer()

    def optimize_model(self, batch_size, gamma=0.99):
        if len(self.memory) < batch_size:
            return
            
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        
        # Reward clipping
        reward = np.clip(reward, -1, 1)
        
        # Compute Q(s_t, a) ,  for the actions which has been taken
        state_action_values = self.policy_net(torch.FloatTensor(state)).gather(1, torch.LongTensor(action))
        
        
        next_state_values = self.target_net(torch.FloatTensor(next_state)).max(1)[0].detach()
        
        # if done , stop the episode
        expected_state_action_values = torch.FloatTensor(reward) + \
                                     (gamma * next_state_values * (1 - torch.FloatTensor(done)))
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

class FrameStacker:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = []
        
    def push(self, frame):
        if len(self.frames) >= self.num_frames:
            self.frames.pop(0)
        self.frames.append(frame)
        
    def get_state(self): # duplicating the last frame to make it 4 frames
        while len(self.frames) < self.num_frames:
            self.frames.append(self.frames[-1])
        return np.stack(self.frames, axis=0)



def evaluate(agent, num_episodes=10, render=False):
    """
    Evaluate the agent's performance and track Q-value predictions
    """
    env = gym.make('ALE/Breakout-v5', render_mode="human" if render else None)
    frame_stacker = FrameStacker()
    
    episode_rewards = []
    q_values = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        frame = PreProcessing.preprecess(obs)
        frame_stacker.push(frame)
        
        episode_reward = 0
        episode_q_values = []
        done = False
        
        while not done:
            state = frame_stacker.get_state()
            
            # Get Q-values for all actions
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0)
                q_value = agent.policy_net(state_tensor)
                episode_q_values.append(q_value.max().item())  # Store maximum Q-value
                action = q_value.max(1)[1].item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            frame = PreProcessing.preprecess(obs)
            frame_stacker.push(frame)
        
        episode_rewards.append(episode_reward)
        q_values.append(np.mean(episode_q_values))
        
        print(f"Evaluation Episode {episode + 1}")
        print(f"  Reward: {episode_reward}")
        print(f"  Average Q-Value: {np.mean(episode_q_values):.2f}")
        print(f"  Max Q-Value: {np.max(episode_q_values):.2f}")
    
    env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_q_value': np.mean(q_values),
        'std_q_value': np.std(q_values)
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
        self.rewards.append(reward)
        self.q_values.append(q_value)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
    
    def get_metrics(self):
        return {
            'last_100_mean_reward': np.mean(self.rewards[-100:]),
            'last_100_mean_q': np.mean(self.q_values[-100:]),
            'last_100_mean_loss': np.mean(self.losses[-100:]),
            'current_epsilon': self.epsilons[-1] if self.epsilons else None
        }






