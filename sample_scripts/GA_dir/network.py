import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from config import elitism_pct, mutation_prob, weights_mutate_power, device
n_squre = 2**8

class Network(nn.Module):

    def __init__(self, env):
        super(Network, self).__init__()
        
        self.num_kind = env.num_kind
        
        # ダミーを入れて、変化を把握
        dots_kind_matrix = np.full((env.num_vertical,env.num_horizontal),0)
        color_mat_3d = self.generate_3d_color_mat(dots_kind_matrix)
        color_mat_3d = torch.tensor(color_mat_3d, dtype=torch.float32, device=device)
        color_mat_3d = torch.flatten(color_mat_3d.unsqueeze(0))
        
        linear_input_size = color_mat_3d.size(dim=0)
        
        self.layer1 = nn.Linear(linear_input_size, n_squre)
        # self.layer2 = nn.Linear(n_squre, n_squre)
        self.layer3 = nn.Linear(n_squre, 1)
        
        self.all_layers = [self.layer1, self.layer3]
        for ii in self.all_layers:
            nn.init.normal(ii.weight, mean=0,std=0.1)

    def forward(self, dots_kind_matrix):
        # dots_kind_matrixはnumpy.array
        
        color_mat_3d = self.generate_3d_color_mat(dots_kind_matrix)
        color_mat_3d = torch.tensor(color_mat_3d, dtype=torch.float32, device=device)
        color_mat_3d = torch.flatten(color_mat_3d.unsqueeze(0))
        
        x = self.layer1(color_mat_3d)
        x = F.relu(x)
        # x = self.layer2(x)
        # x = F.relu(x)
        x = self.layer3(x)
        return x
    
    def generate_3d_color_mat(self, dots_kind_matrix):
        color_mat_3d = []
        for ii in range(self.num_kind):
            each_color_mat = dots_kind_matrix == ii + 1
            color_mat_3d.append(each_color_mat)
        color_mat_3d = np.array(color_mat_3d)
                
        return color_mat_3d