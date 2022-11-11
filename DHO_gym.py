import numpy as np

class TwoDimArrayMap:
    def __init__(self):
        self.states = np.zeros([10, 10])
        self.state = 0
        self.observation_space_dim = 100
        self.action_space_dim = 4
        
    def mazation(self):
        for i in range(1,9):
            for j in range(1,9):
                self.states[i][j] = np.random.randint(0, 2)
        return self
    
    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0 and self.state % 10 != 9 and self.state+1 <=99:           # 오른쪽 한 칸
            if (self.states[(self.state+1) // 10][(self.state+1) % 10]) == 0:         
                self.state = self.state + 1
        elif action == 1 and self.state % 10 != 0 and self.state-1 >= 0:         # 왼쪽 한 칸
            if (self.states[(self.state-1) // 10][(self.state-1) % 10]) == 0:
                self.state = self.state - 1
        elif action == 2 and self.state <=89:                                    # 아래 한 칸
            if (self.states[(self.state+10) // 10][(self.state+10) % 10]) == 0:
                self.state = self.state + 10 
        elif action == 3 and self.state >= 10:                                   # 위 한 칸
            if (self.states[(self.state-10) // 10][(self.state-10) % 10]) == 0:
                self.state = self.state - 10  
        
        if self.state == 99:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        
        return self.state, reward, done