import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from env.tictactoe_env import TicTacToeEnv
import random
from collections import deque

env = TicTacToeEnv()
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class QNetwork(nn.Module):
    def __init__(self, state_size=9,action_size=9, hidden_size=10):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


episodes = 200000
max_steps = 200
gamma = 0.99

# Exploration parameters
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001


hidden_size = 256
learning_rate = 0.0001


memory_size = 100
batch_size = 20
pretrain_length = batch_size



model = QNetwork(hidden_size=hidden_size)

optimizer = optim.Adam(model.parameters(), learning_rate)
loss_fn = torch.nn.MSELoss(size_average=False)


env.reset()
state = env.reset()

memory = Memory(max_size=memory_size)

for ii in range(pretrain_length):

    action = random.randint(0,8)
    next_state, reward, done = env.step(action)

    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))
        env.reset()
    else:
        memory.add((state, action, reward, next_state, done))
        state = next_state


rewards_list = []


draw = 0
win = 0
lose = 0
illegal = 0

step = 0
for ep in range(1, episodes):
    total_reward = 0
    t = 0
    done = False
    while not done:
        step += 1

        explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
        if explore_p > np.random.rand():
            action = random.randint(0, 8)
        else:
            action = model(Variable(FloatTensor([state]), volatile=True)).data.max(1)[1]
            action = action.numpy()[0]

        next_state, reward, done = env.step(action)

        total_reward += reward
        if(reward == env.draw):
            draw += 1
        if(reward == env.win):
            win += 1
        if(reward == env.lose):
            lose += 1
        if(reward == env.illegal):
            illegal += 1


        if done:
            next_state = np.zeros(state.shape)
            t = max_steps
            rewards_list.append((ep, total_reward))
            memory.add((state, action, reward, next_state, done))
            env.reset()

        else:
            memory.add((state, action, reward, next_state, done))
            state = next_state
            t += 1

        batch = memory.sample(batch_size)
        states = Variable(torch.from_numpy(np.array([each[0] for each in batch]))).type(FloatTensor)
        actions = Variable(torch.from_numpy(np.array([each[1] for each in batch]))).type(LongTensor)
        rewards = Variable(torch.from_numpy(np.array([each[2] for each in batch]))).type(FloatTensor)
        next_states = Variable(FloatTensor(np.array([each[3] for each in batch])), volatile=True)
        dones = Variable(torch.FloatTensor([each[4] for each in batch]))

        target_Qs = model(next_states)
        max = target_Qs.max(1)[0]
        targets = rewards + gamma * max * (1-dones)


        result = model(states)
        current_q_values = model(states).gather(1, actions.view(-1, 1))

        loss = loss_fn(current_q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if ep % 10 == 0:
        print('Episode: {}'.format(ep),
                  'Total reward: {}'.format(total_reward),
                  'draw: {}'.format(draw),
                  'win: {}'.format(win),
                  'lose: {}'.format(lose),
                  'illegal: {}'.format(illegal),
                  'Explore {:.4f}'.format(explore_p))

        draw = 0
        win = 0
        lose = 0
        illegal = 0
