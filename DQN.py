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

BATCH_SIZE          = 128
GAMMA               = 0.995
EPS_START           = 0.9
EPS_END             = 0.05
EPS_DECAY           = 5000
RANDOM_ACTION_PROB  = 0.3
RANDOM_GOAL         = True

X_SIZE          = 18
Y_SIZE          = 18

STATE_DIM       = 2
GOAL_DIM        = 2        
ACTION_DIM      = 4

NUM_EPISODES    = 200
TIME_LIMIT      = 600

TARGET_UPDATE   = 30

SAVE            = NUM_EPISODES//10

# Goal conditioned RL? and other Global variables
GoalCon = True
steps_done = 0

# logging
now_day = datetime.datetime.now().strftime("%m-%d")
now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")

# networks
if GoalCon:
    Q_net = Networks.QNET((STATE_DIM + GOAL_DIM), ACTION_DIM).to('cuda')
    target_Q_net = Networks.QNET((STATE_DIM + GOAL_DIM), ACTION_DIM).to('cuda')
else:
    Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to('cuda')
    target_Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to('cuda')
target_Q_net.load_state_dict(Q_net.state_dict())
target_Q_net.eval()

# optimizer
optimizer = optim.RMSprop(Q_net.parameters())
memory = ReplayMemory(10000)
QnetToCell = DHO_utils.QnetToCell(X_SIZE, Y_SIZE)

def select_action(state):
    global steps_done
    sample = random.random()    # 0 ~ 1 사이의 랜덤한 값
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = np.clip(eps_threshold, RANDOM_ACTION_PROB, 0.9)
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
    
    env = DHO_gym.TwoDimCoordinationMap(X_SIZE, Y_SIZE)
    env.SimpleAntMazation()
    path = f'./results_DQN/results_DQN_{now_day}_{X_SIZE, Y_SIZE}_{GAMMA}_NE_{NUM_EPISODES}_TSL_{TIME_LIMIT}_{GoalCon}_{TARGET_UPDATE}_{RANDOM_ACTION_PROB}_{RANDOM_GOAL}/result_{now}'
    writer = SummaryWriter(f'{path}/tensorboard_{now}')
    if not os.path.isdir(path):
        os.makedirs(path)
    np.savetxt(f'{path}/SimpleMaze_table.txt', env.maze, fmt='%d')
    np.savetxt(f'{path}/SimpleMaze_Reward_table.txt', env.reward_states, fmt='%d')
    
    env.randomGoal = RANDOM_GOAL
    for i_episode in range(NUM_EPISODES):
        state = env.reset(GoalCon)
        np.savetxt(f'{path}/SimpleMaze_Reward_table_reset.txt', env.reward_states, fmt='%d')
        state = torch.tensor(state, device='cuda', dtype=torch.float32)
        
        for t in range(1, TIME_LIMIT+1):   # t가 1부터 1씩 증가
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
                break
            
        if loss is not None:
            writer.add_scalar('success_rate/train', int(done), i_episode)
            writer.add_scalar('step_per_episode/train', t, i_episode)
            writer.add_scalar('Loss/train', loss, i_episode)
        
        if i_episode % 10 == 0:
            print(f"episode of {i_episode} is done with {t} steps. It should be less than {X_SIZE*Y_SIZE}")
            
            if i_episode % SAVE == 0:
                V_table, Action_table = QnetToCell.FillGridByQnet(Q_net, env, GoalCon)
                now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
                print(f"episode of {i_episode} is saved with {t} steps. V_table and Action table")
                
                if not os.path.isdir(path+'/V_table_train') or not os.path.isdir(path+'/Action_table_train'):
                    os.makedirs(path+'/V_table_train')
                    os.makedirs(path+'/Action_table_train')
                    
                np.savetxt(f'{path}/V_table_train/V_table_{i_episode}_{now}.txt', V_table, fmt='%.3f')
                np.savetxt(f'{path}/Action_table_train/Action_table_{i_episode}_{now}.txt', Action_table, fmt='%d')
        
    
    #### test ####
    env.randomGoal = False
    for i_episode in range(int(NUM_EPISODES)):
        state = env.reset(GoalCon)
        state = torch.tensor(state, device='cuda', dtype=torch.float32)
        
        for t in range(1, TIME_LIMIT+1):   # t가 1부터 1씩 증가
            action = select_action(state)
            next_state, reward, done = env.step(action.item())  # action.item()은 action의 값
            reward = torch.tensor(reward,dtype=torch.float32 ,device='cuda')
            next_state = torch.tensor(next_state, device='cuda', dtype=torch.float32)
            state = next_state
            if done:
                break
        writer.add_scalar('success_rate/test', int(done), i_episode)
        writer.add_scalar('step_per_episode/test', t, i_episode)
        
        if i_episode % SAVE == 0:
            print(f"Test episode of {i_episode} is done with {t} steps")
            V_table, Action_table = QnetToCell.FillGridByQnet(Q_net, env, GoalCon)
            now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
            
            if not os.path.isdir(path+'/V_table_test') or not os.path.isdir(path+'/Action_table_test'):
                    os.makedirs(path+'/V_table_test')
                    os.makedirs(path+'/Action_table_test')
            np.savetxt(f'{path}/V_table_test/V_table_test_{now}.txt', V_table, fmt='%.3f')
            np.savetxt(f'{path}/Action_table_test/Action_table_test_{now}.txt', Action_table, fmt='%d')
    writer.close()
    print(f'{path} is done')
    print('Complete')