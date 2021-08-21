import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

mu = torch.tensor([ 1.0610])
std = torch.tensor([0.0688])
dist = Normal(mu, std)
while True:
    a = dist.sample()
    prob= dist.log_prob(a)
    p = prob.exp()
pass
