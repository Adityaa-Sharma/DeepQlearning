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
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(torch.FloatTensor(state)).gather(1, torch.LongTensor(action))
        
        # Compute V(s_{t+1}) for all next states
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
        
    def get_state(self):
        while len(self.frames) < self.num_frames:
            self.frames.append(self.frames[-1])
        return np.stack(self.frames, axis=0)

def train(episodes, batch_size=32):
    env = gym.make('ALE/Breakout-v5')
    agent = DQNAgent(input_shape=(4, 84, 84))  # 4 stacked frames
    frame_stacker = FrameStacker()
    
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = (epsilon - epsilon_min) / 1000000  # Decay over 1M frames
    
    for episode in range(episodes):
        obs, _ = env.reset()
        frame = PreProcessing.preprecess(obs)
        frame_stacker.push(frame)
        
        total_reward = 0
        done = False
        
        while not done:
            state = frame_stacker.get_state()
            action = agent.policy_net.act(state, epsilon)
            
            # Frame skipping (k=4)
            reward = 0
            for _ in range(4):
                obs, r, terminated, truncated, _ = env.step(action)
                reward += r
                if terminated or truncated:
                    done = True
                    break
                    
            frame = PreProcessing.preprecess(obs)
            frame_stacker.push(frame)
            next_state = frame_stacker.get_state()
            
            # Store transition in memory
            agent.memory.push(state, action, reward, next_state, done)
            
            # Optimize model
            agent.optimize_model(batch_size)
            
            # Update target network periodically
            if episode % 10 == 0:  # Update every 10 episodes
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                
            total_reward += reward
            epsilon = max(epsilon_min, epsilon - epsilon_decay)
            
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")






