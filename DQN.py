# code source from https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html
# DHO_gym is handmade by me

import os
import math
import random
import datetime
import numpy as np

from itertools import count

import env.DHO_gym as DHO_gym
import util.DHO_utils as DHO_utils
import model.Networks as Networks
from util.ReplayMemory import ReplayMemory

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/DQN_{}'.format(datetime.datetime.now().strftime("%m%d-%H%M%S")))

BATCH_SIZE      = 128
GAMMA           = 0.9
EPS_START       = 0.9
EPS_END         = 0.05
EPS_DECAY       = 5000
TARGET_UPDATE   = 10
X_SIZE          = 20
Y_SIZE          = 20

STATE_DIM       = 2
ACTION_DIM      = 4

NUM_EPISODES    = 5000
TIME_LIMIT      = 500

ENDSTATE        = torch.tensor([X_SIZE-1, 0], device='cuda')

SAVE            = NUM_EPISODES/10

Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to('cuda')
target_Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to('cuda')
target_Q_net.load_state_dict(Q_net.state_dict())
target_Q_net.eval()

optimizer = optim.RMSprop(Q_net.parameters())
memory = ReplayMemory(10000)
QnetToCell = DHO_utils.QnetToCell(X_SIZE, Y_SIZE)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()    # 0 ~ 1 사이의 랜덤한 값
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = np.clip(eps_threshold, 0.3, 0.8)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return Q_net(state).max(0)[1]
    else:
        return torch.tensor(random.randrange(ACTION_DIM), device='cuda', dtype=torch.int64)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    batch = memory.Transition(*zip(*transitions))
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action).unsqueeze(-1)
    reward_batch = torch.stack(batch.reward)
    non_final_next_states = torch.stack(batch.next_state)
    mask_batch = torch.stack(batch.mask)
    
    Q_values = Q_net(state_batch).gather(1, action_batch)
    
    with torch.no_grad():
        next_state_Q_values_array = target_Q_net(non_final_next_states).max(1)[0]
        expected_Q_values_array = (next_state_Q_values_array.mul(mask_batch) * GAMMA) + reward_batch
    
    criterion = nn.MSELoss()
    loss = criterion(Q_values, expected_Q_values_array.unsqueeze(-1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in Q_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss

if __name__ == '__main__':
    episode_durations = []

    env = DHO_gym.TwoDimCoordinationMap(X_SIZE, Y_SIZE)
    env.SimpleAntMazation()
    
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    path = f'./results_DQN/result_{now}'
    
    if not os.path.isdir(path):
        os.makedirs(path)
    np.savetxt(f'{path}/SimpleMaze_table.txt', env.states, fmt='%d')
    np.savetxt(f'{path}/SimpleMaze_Reward_table.txt', env.reward_states, fmt='%d')
    
    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        state = torch.tensor(state, device='cuda', dtype=torch.float32)
        
        for t in count():   # t가 1부터 1씩 증가
            action = select_action(state)
            next_state, reward, done = env.step(action.item())  # action.item()은 action의 값
            reward = torch.tensor(reward,dtype=torch.float32 ,device='cuda')
            next_state = torch.tensor(next_state, device='cuda', dtype=torch.float32)
            
            memory.push(state, action, next_state, reward, torch.tensor(1-int(done), device='cuda', dtype=torch.float32))
            state = next_state

            loss = optimize_model()
            if t % TARGET_UPDATE == 0:
                target_Q_net.load_state_dict(Q_net.state_dict())
            if done:
                episode_durations.append(t + 1)
                break
            if t > TIME_LIMIT-1:
                break
        if loss is not None:
            writer.add_scalar('success/train', int(done), i_episode)
            writer.add_scalar('step_per_episode/train', t, i_episode)
            writer.add_scalar('Loss/train', loss, i_episode)
            writer.flush()
        
        if i_episode % TARGET_UPDATE == 0:
            print(f"episode of {i_episode} is done with {t} steps")
            if i_episode % SAVE == 0:
                V_table, Action_table = QnetToCell.FillGridByQnet(Q_net)
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                np.savetxt(f'{path}/V_table_{now}.txt', V_table, fmt='%.3f')
                np.savetxt(f'{path}/Action_table_{now}.txt', Action_table, fmt='%d')
    writer.close()
    print('Complete')