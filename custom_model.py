import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class CustomModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7 + 1, 512)  # +1 for additional state information
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x, additional_state):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, additional_state), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = CustomModel(state_size, action_size)
        self.target_model = CustomModel(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state, additional_state = state
        state = torch.FloatTensor(state).unsqueeze(0)
        additional_state = torch.FloatTensor(additional_state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state, additional_state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state, additional_state = state
            next_state, next_additional_state = next_state
            state = torch.FloatTensor(state).unsqueeze(0)
            additional_state = torch.FloatTensor(additional_state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            next_additional_state = torch.FloatTensor(next_additional_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state, next_additional_state)).item()
            target_f = self.model(state, additional_state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state, additional_state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def step(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        if len(self.memory) > batch_size:
            self.replay(batch_size)
        self.update_target_model()
