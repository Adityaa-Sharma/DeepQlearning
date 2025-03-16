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
    

def DQN_Target_Network():
    pass


def PreProcessing():
    # This will make the input as 210x160x3 to 84x84x4
    def preprecess(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        gray=gray/255.0
        return torch.FloatTensor(gray).unsqueeze(0)
        

    