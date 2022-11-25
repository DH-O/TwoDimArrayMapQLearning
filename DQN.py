# code source from https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html
# DHO_gym is handmade by me

import os
import math
import random
import argparse
import datetime
import numpy as np

import env.DHO_gym as DHO_gym
import util.DHO_utils as DHO_utils
import model.Networks as Networks
from util.ReplayMemory import ReplayMemory

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

RM_SIZE = 1000000
BATCH_SIZE          = 256
GAMMA               = 0.995
EPS_START           = 0.9
EPS_END             = 0.05
EPS_DECAY           = 500
RANDOM_ACTION_PROB  = 0.1
RANDOM_GOAL         = False

X_SIZE          = 10
Y_SIZE          = 10

STATE_DIM       = 2
GOAL_DIM        = 2        
ACTION_DIM      = 4

NUM_EPISODES    = 600
TIME_LIMIT      = 200
TEST_EPISODES = 100

TARGET_UPDATE   = 500

SAVE            = NUM_EPISODES//10

# Goal conditioned RL? and other Global variables
GOAL_CON = True
TEST = False

steps_done = 0

parser = argparse.ArgumentParser(description='DQN')

parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

# logging
now_day = datetime.datetime.now().strftime("%m-%d")
now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")

path = f'./results_DQN/results_DQN_{now_day}_{X_SIZE, Y_SIZE}_{GAMMA}_NE_{NUM_EPISODES}_TSL_{TIME_LIMIT}_{GOAL_CON}_{TARGET_UPDATE}_{RANDOM_ACTION_PROB}_{RANDOM_GOAL}_{RM_SIZE}/result_{now}'
writer = SummaryWriter(f'{path}/tensorboard_{now}')

# networks
if GOAL_CON:
    Q_net = Networks.QNET((STATE_DIM + GOAL_DIM), ACTION_DIM).to(device)
    target_Q_net = Networks.QNET((STATE_DIM + GOAL_DIM), ACTION_DIM).to(device)
else:
    Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to(device)
    target_Q_net = Networks.QNET(STATE_DIM, ACTION_DIM).to(device)
target_Q_net.load_state_dict(Q_net.state_dict())
target_Q_net.eval()

# optimizer
optimizer = optim.RMSprop(Q_net.parameters())
memory = ReplayMemory(RM_SIZE)
QnetToCell = DHO_utils.QnetToCell(X_SIZE, Y_SIZE)

def select_action(state):
    global steps_done
    if TEST:
        return Q_net(state).max(0)[1]
    else:
        sample = random.random()    # 0 ~ 1 사이의 랜덤한 값
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        eps_threshold = np.clip(eps_threshold, RANDOM_ACTION_PROB, 0.9)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return Q_net(state).max(0)[1]
        else:
            return torch.tensor(random.randrange(ACTION_DIM), device=device, dtype=torch.int64)

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
    
    #### Maze logging
    if not os.path.isdir(path):
        os.makedirs(path)
    np.savetxt(f'{path}/SimpleMaze_table.txt', env.maze, fmt='%d')
    np.savetxt(f'{path}/SimpleMaze_Reward_table.txt', env.reward_states, fmt='%d')
    
    for i_episode in range(NUM_EPISODES+1):
        
        env.randomGoal = RANDOM_GOAL
        TEST = False
        
        state = env.reset(GOAL_CON)
        np.savetxt(f'{path}/SimpleMaze_Reward_table_reset.txt', env.reward_states, fmt='%d')
        
        state = torch.tensor(state, device=device, dtype=torch.float32)
        
        for t in range(1, TIME_LIMIT+1):   # t가 1부터 1씩 증가
            action = select_action(state)
            next_state, reward, done = env.step(action.item())  # action.item()은 action의 값
            reward = torch.tensor(reward,dtype=torch.float32 ,device=device)
            next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
            
            memory.push(state, action, next_state, reward, torch.tensor(1-int(done), device=device, dtype=torch.float32))
            state = next_state
            
            loss = optimize_model()
            
            if steps_done % TARGET_UPDATE == 0:
                target_Q_net.load_state_dict(Q_net.state_dict())
            if done:
                break
            
        if loss is not None:
            writer.add_scalar('success_rate/train', int(done), i_episode)
            writer.add_scalar('steps_per_episode/train', t, i_episode)
            writer.add_scalar('Loss/train', loss, i_episode)
        
        if i_episode % 10 == 0:
            if t != TIME_LIMIT:    
                print(f"episode of {i_episode} is done with {t} steps. It should be less than {X_SIZE*Y_SIZE}")
            
            #### test ####
            env.randomGoal = False
            TEST = True
            done_sum = 0
            steps_sum = 0
            
            for _ in range(TEST_EPISODES):
                state = env.reset(GOAL_CON)
                state = torch.tensor(state, device=device, dtype=torch.float32)
                
                for test_t in range(1, TIME_LIMIT+1):   # t가 1부터 1씩 증가
                    action = select_action(state)
                    next_state, _, done = env.step(action.item())  # action.item()은 action의 값
                    next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
                    state = next_state
                    if done:
                        break
                done_sum += int(done)
                steps_sum += test_t
                
            writer.add_scalar('success_rate/test', done_sum/TEST_EPISODES, i_episode)
            writer.add_scalar('steps_per_episode/test', steps_sum/TEST_EPISODES, i_episode)
            if test_t != TIME_LIMIT:    
                print(f"----Test episodes of {i_episode} is done with {steps_sum/TEST_EPISODES} steps")
            V_table, Action_table = QnetToCell.FillGridByQnet(Q_net, env, GOAL_CON, device)
            
            
            if not os.path.isdir(path+'/V_table_test') or not os.path.isdir(path+'/Action_table_test'):
                    os.makedirs(path+'/V_table_test')
                    os.makedirs(path+'/Action_table_test')
            
            now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
            np.savetxt(f'{path}/V_table_test/V_table_test_{now}.txt', V_table, fmt='%.3f')
            np.savetxt(f'{path}/Action_table_test/Action_table_test_{now}.txt', Action_table, fmt='%d')
            
            if i_episode % SAVE == 0:
                V_table, Action_table = QnetToCell.FillGridByQnet(Q_net, env, GOAL_CON, device)
                if t != TIME_LIMIT:    
                    print(f"--------{i_episode} is saved with {t} steps. V_table and Action table")
                
                if not os.path.isdir(path+'/V_table_train') or not os.path.isdir(path+'/Action_table_train'):
                    os.makedirs(path+'/V_table_train')
                    os.makedirs(path+'/Action_table_train')
                
                now = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
                np.savetxt(f'{path}/V_table_train/V_table_{i_episode}_{now}.txt', V_table, fmt='%.3f')
                np.savetxt(f'{path}/Action_table_train/Action_table_{i_episode}_{now}.txt', Action_table, fmt='%d')    

    writer.close()
    print(f'{path} is done')
    print('Complete')