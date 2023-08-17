import numpy as np
import json
import pandas as pd
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import warnings
from pandas.errors import SettingWithCopyWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from model import * # DRL4TSP, Encoder, StateCritic, Critic
from tsp import *
from helper import *
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


#### trainset
train_id = list(train_route)

#### train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

random.shuffle(train_id)

m = 4890 
batch_size = 128
max_grad_norm = 2

train_data = TSPDataset(train_id[:m], 'train')
# valid_data = TSPDataset(train_id[m:], 'train')

actor = DRL4TSP(static_size=train_data.num_zone, 
                dynamic_size=1,
                hidden_size=256,
                mask_fn=update_mask,
                num_layers=2,
                dropout=0.2
                ).to(device)

critic = StateCritic(train_data.num_zone, 1, 256).to(device)

actor_optim = optim.Adam(actor.parameters(), lr=1e-4)
critic_optim = optim.Adam(critic.parameters(), lr=1e-4)

train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
# valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

best_params = None
best_reward = np.inf

plot_loss, plot_reward, route_score = [], [], []

for epoch in tqdm(range(30)):

    actor.train()
    critic.train()

    losses, rewards, critic_rewards = [], [], []

    for batch_idx, batch in enumerate(train_loader):

        static, dynamic, x0, actual_seq, route_id = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        tour_indices, tour_logp = actor(static, dynamic, x0)

        ## ---- different reward ----        
        # use travel time as reward
        reward0 = cal_time(tour_indices, route_id, 'train')
        # reward0 = customize_time_mat(tour_indices, route_id, 'train')
        # reward0 = name_later(tour_indices, route_id, '3opt', 'train')

        reward = torch.tensor(reward0, dtype=torch.float32)

        # Query the critic for an estimate of the reward
        critic_est = critic(static, dynamic).view(-1)

        advantage = (reward - critic_est)
        
        actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
        critic_loss = torch.mean(advantage ** 2)

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_optim.step()

        critic_rewards.append(torch.mean(critic_est.detach()).item())
        rewards.append(torch.mean(reward.detach()).item())
        losses.append(torch.mean(actor_loss.detach()).item())

        score_stage2 = name_later(tour_indices, route_id, '3opt', 'train')
        s2 = np.mean(score_stage2)

    plot_loss.append(np.mean(losses))
    plot_reward.append(np.mean(rewards))
    route_score.append(s2)

#### eval

valid_data = TSPDataset(train_id[m:], 'train')
valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

val_reward = []
num_epoch = 10

for epoch in tqdm(range(num_epoch)):
    actor.eval()

    for batch_idx, batch in enumerate(valid_loader):

        static, dynamic, x0, actual_seq, route_id = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        score_stage2 = name_later(tour_indices, route_id, '3opt', 'train') 
        s2 = np.mean(score_stage2)

    val_reward.append(s2)

