import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from envs.tictactoe_env import TicTacToeEnv
import random
import collections

env = TicTacToeEnv(illegal = -1, win= 1, draw= 0, lose = -1 ,default = -0.1)
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=0)
        return x

input_size = 9
hidden_size = 100
output_size = 9
batch_size = 100
learning_rate = 1e-4

log_probs, rewards = [], []
results = collections.deque()
draw, win, lose, illegal = 0, 0, 0, 0
batch_size = 20

policy = Policy(input_size, hidden_size, output_size)
optimizer = torch.optim.RMSprop(policy.parameters(), lr=learning_rate)

state = env.reset()

episode = 0
total_reward = 0
while True:
  state = env.reset()
  done = False
  step = 0
  episode += 1
  while not done:
    probs = policy(Variable(FloatTensor(state)))
    m = Categorical(probs)
    action = m.sample()
    log_probs.append(m.log_prob(action))
    state, reward, done = env.step(action.data[0])
    rewards.append(reward)
    step+=1

  total_reward += reward

  if(reward == env.draw):
      draw += 1
  if(reward == env.win):
      win += 1
  if(reward == env.lose):
      lose += 1
  if(reward == env.illegal):
      illegal += 1

  #reward /= float(step)
  #rewards += ([reward] * step)

  if episode % batch_size == 0:

    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    policy_loss = []

    for log_prob, reward in zip(log_probs, rewards):
      policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    rewards = []
    log_probs = []


  if episode % 100 == 0:
      print('Episode: {}'.format(episode),
                'Total reward: {}'.format(total_reward),
                'draw: {}'.format(draw),
                'win: {}'.format(win),
                'lose: {}'.format(lose),
                'illegal: {}'.format(illegal))

      draw = 0
      win = 0
      lose = 0
      illegal = 0
      total_reward = 0



