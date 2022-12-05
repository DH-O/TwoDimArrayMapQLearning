import numpy as np

class TwoDimArrayMap:
    def __init__(self, x_dim, y_dim, action_space_dim = 4):
        self.maze = np.zeros([x_dim, y_dim])
        self.reward_states = np.full_like(np.zeros([x_dim, y_dim]), -1)
        
        self.state = np.array([0, 0])
        self.observation_space_dim = x_dim * y_dim
        self.action_space_dim = action_space_dim
        self.row = len(self.maze)
        self.col = len(self.maze[0])
        
    def mazation(self): # random obstacles in the maze
        object_list = [0, 1]
        for i in range(1,self.row-1):
            for j in range(1,self.col-1):
                self.maze[i][j] = np.random.choice(object_list, 1, p=[0.8, 0.2])
        return self
    
    def SimpleAntMazation(self):   # simple maze having one large wall. Represent wall as 1. At the reward states, wall is -9
        for i in range(self.row):
            for j in range(self.col):
                if (self.row//3) <= i < (2 * self.row//3):
                    if j < self.col * (2/3):
                        self.maze[i][j] = 1
                        self.reward_states[i][j] = -9
        return self
    
    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0 and self.state % self.col != (self.col - 1) and self.state + 1 < self.observation_space_dim:     # 오른쪽 한 칸
            if (self.maze[(self.state + 1) // self.row][(self.state + 1) % self.col]) == 0:         
                self.state = self.state + 1
        elif action == 1 and self.state % self.col != 0 and self.state - 1 >= 0:                                        # 왼쪽 한 칸
            if (self.maze[(self.state - 1) // self.row][(self.state - 1) % self.col]) == 0:
                self.state = self.state - 1
        elif action == 2 and self.state < (self.observation_space_dim - self.row):                                      # 아래 한 칸
            if (self.maze[(self.state + self.row) // self.row][(self.state + self.row) % self.col]) == 0:
                self.state = self.state + self.row 
        elif action == 3 and self.state > (self.row - 1):                                                               # 위 한 칸
            if (self.maze[(self.state - self.row) // self.row][(self.state - self.row) % self.col]) == 0:
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
        self.init_goal = np.array([0, self.col-1])
        self.randomGoal = False
        
    def reset(self, GoalCon, test_case=1):
        self.state = np.array([0, 0])
        if self.randomGoal == True:
            self.goal = np.array([np.random.randint(0, self.row), np.random.randint(0, self.col)])
            while (self.maze[self.goal[0]][self.goal[1]] == 1):
                self.goal = np.array([np.random.randint(0, self.row), np.random.randint(0, self.col)])
            if self.goal[0] == self.init_goal[0] and self.goal[1] == self.init_goal[1]:
                print("Random Goal is the same as the original goal")
            self.reward_states[self.goal[0]][self.goal[1]] = 1
        else:
            self.reward_states[self.goal[0]][self.goal[1]] = -1
            self.goal = self.init_goal
            if test_case == 2:
                # self.goals.append(self.init_goal)
                self.goal = np.array([self.row - 1, self.col - 1])
            elif test_case == 3:
                # self.goals.append(self.init_goal)
                # self.goals.append(np.array([self.row - 1, self.col - 1]))
                self.goal = np.array([self.row - 1, 0])

            self.reward_states[self.goal[0]][self.goal[1]] = 1
        
        if GoalCon:
            self.state = np.concatenate((self.state, self.goal))
        
        return self.state
    
    def step(self, action):
        if action == 0 and self.state[1] < self.col-1:
            if self.maze[self.state[0]][self.state[1]+1] == 0:
                self.state[1] += 1
        elif action == 1 and self.state[1] > 0:
            if self.maze[self.state[0]][self.state[1]-1] == 0:
                self.state[1] -= 1
        elif action == 2 and self.state[0] < self.row-1:
            if self.maze[self.state[0]+1][self.state[1]] == 0:
                self.state[0] += 1
        elif action == 3 and self.state[0] > 0:
            if self.maze[self.state[0]-1][self.state[1]] == 0:
                self.state[0] -= 1
        
        if (self.state[0] == self.goal[0]) and (self.state[1] == self.goal[1]):
            reward = 1
            done = True
        else:
            reward = -1
            done = False
            
        return self.state, reward, done