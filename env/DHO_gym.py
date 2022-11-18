import numpy as np

class TwoDimArrayMap:
    def __init__(self, x_dim, y_dim, action_space_dim = 4):
        self.states = np.zeros([x_dim, y_dim])
        self.reward_states = np.zeros([x_dim, y_dim])
        self.state = np.array([0, 0])
        self.observation_space_dim = self.states.size
        self.action_space_dim = action_space_dim
        self.row = len(self.states)
        self.col = len(self.states[0])
        
    def mazation(self):
        object_list = [0, 1]
        for i in range(1,self.row-1):
            for j in range(1,self.col-1):
                self.states[i][j] = np.random.choice(object_list, 1, p=[0.8, 0.2])
        return self
    
    def SimpleAntMazation(self):
        for i in range(self.row):
            for j in range(self.col):
                if (self.row//3) <= i < (2 * self.row//3):
                    if j < self.col * (2/3):
                        self.states[i][j] = 1
                        self.reward_states[i][j] = 1
        return self
    
    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0 and self.state % self.col != (self.col - 1) and self.state + 1 < self.observation_space_dim:     # 오른쪽 한 칸
            if (self.states[(self.state + 1) // self.row][(self.state + 1) % self.col]) == 0:         
                self.state = self.state + 1
        elif action == 1 and self.state % self.col != 0 and self.state - 1 >= 0:                                        # 왼쪽 한 칸
            if (self.states[(self.state - 1) // self.row][(self.state - 1) % self.col]) == 0:
                self.state = self.state - 1
        elif action == 2 and self.state < (self.observation_space_dim - self.row):                                      # 아래 한 칸
            if (self.states[(self.state + self.row) // self.row][(self.state + self.row) % self.col]) == 0:
                self.state = self.state + self.row 
        elif action == 3 and self.state > (self.row - 1):                                                               # 위 한 칸
            if (self.states[(self.state - self.row) // self.row][(self.state - self.row) % self.col]) == 0:
                self.state = self.state - self.row
        
        if self.state == self.observation_space_dim - self.row:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        
        return self.state, reward, done
    
class TwoDimCoordinationMap(TwoDimArrayMap):
    def __init__(self, x_dim, y_dim, action_space_dim=4):
        super().__init__(x_dim, y_dim, action_space_dim)
        self.state = np.array([0, 0])
        
    def reset(self):
        self.state = np.array([0, 0])
        return self.state
    
    def step(self, action):
        if action == 0 and self.state[1] < self.col-1:
            if self.states[self.state[0]][self.state[1]+1] == 0:
                self.state[1] += 1
        elif action == 1 and self.state[1] > 0:
            if self.states[self.state[0]][self.state[1]-1] == 0:
                self.state[1] -= 1
        elif action == 2 and self.state[0] < self.row-1:
            if self.states[self.state[0]+1][self.state[1]] == 0:
                self.state[0] += 1
        elif action == 3 and self.state[0] > 0:
            if self.states[self.state[0]-1][self.state[1]] == 0:
                self.state[0] -= 1
        
        if (self.state[0] == self.row-1) and (self.state[1] == 0):
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        return self.state, reward, done
    
    def SimpleAntMazation(self):
        for i in range(self.row):
            for j in range(self.col):
                if (i == self.row-1) and (j == 0):
                    self.reward_states[i][j] = 2
                if (self.row//3) <= i < (2 * self.row//3):
                    if j < self.col * (2/3):
                        self.states[i][j] = 1
                        self.reward_states[i][j] = 1