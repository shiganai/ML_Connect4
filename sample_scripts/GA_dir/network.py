import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from config import elitism_pct, mutation_prob, weights_mutate_power, device
# n_squre = 2**7
n_squre = 200

class Network(nn.Module):

    def __init__(self, env):
        super(Network, self).__init__()
        
        self.num_kind = env.num_kind
        
        full_kernel_size = max([env.num_vertical,env.num_horizontal])
        
        self.conv2d_each_color = nn.Conv2d(in_channels=1, out_channels=1, \
                                           kernel_size=full_kernel_size, \
                                           padding = int(full_kernel_size/2) )
                                           
        self.conv2d_not_each_color = nn.Conv2d(in_channels=1, out_channels=1, \
                                                kernel_size=full_kernel_size, \
                                                padding = int(full_kernel_size/2) )
                                                
        self.conv2d_empty = nn.Conv2d(in_channels=1, out_channels=1, \
                                        kernel_size=full_kernel_size, \
                                        padding = int(full_kernel_size/2) )
        
        # ダミーを入れて、変化を把握
        dots_kind_matrix = np.full((env.num_vertical,env.num_horizontal),0)
        color_mat_3d = self.generate_linear_input(dots_kind_matrix)
        
        linear_input_size = color_mat_3d.size(dim=0)
        
        self.layer1 = nn.Linear(linear_input_size, n_squre)
        self.layer2 = nn.Linear(n_squre, n_squre)
        self.layer3 = nn.Linear(n_squre, 1) # 何連鎖できそうなのかの期待値を返すイメージ
        
        self.all_layers = [self.layer1, self.layer2, self.layer3, \
                           self.conv2d_each_color, self.conv2d_not_each_color, self.conv2d_empty]
        for ii in self.all_layers:
            nn.init.normal(ii.weight, mean=0,std=0.12)

    def forward(self, dots_kind_matrix):
        # dots_kind_matrixはnumpy.array
        
        color_mat_3d = self.generate_linear_input(dots_kind_matrix)
        color_mat_3d = torch.tensor(color_mat_3d, dtype=torch.float32, device=device).unsqueeze(0)
        color_mat_3d = torch.flatten(color_mat_3d)
        
        x = self.layer1(color_mat_3d)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer2(x)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer3(x)
        return x
    
    def generate_linear_input(self, dots_kind_matrix):
        dots_kind_matrix = dots_kind_matrix[0:-2,:] # trimming
        for ii in range(self.num_kind):
            each_color_mat = (dots_kind_matrix == ii + 1) * 1.0
            empty = (dots_kind_matrix == 0) * 1.0
            not_each_color_mat = np.logical_not(each_color_mat + empty > 0) * 1.0
            
            each_color_mat = torch.tensor(each_color_mat, dtype=torch.float32, device=device).unsqueeze(0)
            each_color_mat = self.conv2d_each_color(each_color_mat) # つながりそうかの評価
            
            empty = torch.tensor(empty, dtype=torch.float32, device=device).unsqueeze(0)
            empty = self.conv2d_empty(empty) # つながりそうかの評価 # 空のところはつながりやすい？
            
            each_color_mat += empty
            
            not_each_color_mat = torch.tensor(not_each_color_mat, dtype=torch.float32, device=device).unsqueeze(0)
            not_each_color_mat = self.conv2d_not_each_color(not_each_color_mat) # つながらなさそうかの評価
            
            each_color_mat -= not_each_color_mat
            
            each_color_mat = each_color_mat.unsqueeze(3) # 4次元化
            
            if ii == 0:
                color_mat_3d = each_color_mat # I don't know empty initialization of tensor, so initialize at the first loop
            else:
                color_mat_3d = torch.cat((color_mat_3d, each_color_mat), dim=3)
        color_mat_3d = torch.flatten(color_mat_3d)
        return color_mat_3d