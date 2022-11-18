import numpy as np
import torch

class OneLineToCell:
    def __init__(self, x_dim, y_dim):
        self.states = np.zeros([x_dim, y_dim])
        self.state = 0
        self.observation_space_dim = self.states.size
        self.row = len(self.states)
        self.col = len(self.states[0])
    
    def FillGridByOneLineArray(self, array):
        for i in range(self.row):
            for j in range(self.col):
                self.states[i][j] = array[i*self.row + j]
        return self.states

class QnetToCell:
    def __init__(self, x_dim, y_dim):
        self.V_states = np.zeros([x_dim, y_dim])
        self.Action = np.zeros([x_dim, y_dim])
        self.row = len(self.V_states)
        self.col = len(self.V_states[0])
        
    def FillGridByQnet(self, Qnet):
        for i in range(self.row):
            for j in range(self.col):
                torch_state = torch.tensor(np.array([i, j]), dtype=torch.float32, device='cuda')
                self.V_states[i][j] = Qnet(torch_state).max(0)[0].item()
                self.Action[i][j] = Qnet(torch_state).max(0)[1].item()
        return self.V_states, self.Action