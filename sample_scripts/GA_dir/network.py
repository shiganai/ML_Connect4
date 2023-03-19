import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from config import elitism_pct, mutation_prob, weights_mutate_power, device
n_squre = 2**7

class Network(nn.Module):

    def __init__(self, env):
        super(Network, self).__init__()
        
        self.num_kind = env.num_kind
        self.max_kernel_size = max([env.num_horizontal, env.num_vertical])
        
        # 各色に対して共通のフィルタ
        self.conv_each_color_near = nn.Conv2d(in_channels=1, out_channels=1, \
                                             kernel_size=3, padding=1)
        self.conv_each_color_far = nn.Conv2d(in_channels=1, out_channels=1, \
                                            kernel_size=self.max_kernel_size, \
                                            padding=int(self.max_kernel_size/2))
        
        # 3次元的に並べた全色の行列に対してのフィルタ
        # 各色に対して出力するから out_channels=self.num_kind
        # 全色を考慮するから kernel_size=self.num_kind
        self.conv_all_color = nn.Conv3d(in_channels=1, out_channels=self.num_kind, \
                                            kernel_size=self.num_kind, \
                                            padding= int(self.num_kind/2 + 0.5) )
        
        # ダミーを入れて、変化を把握
        dots_kind_matrix = np.full((env.num_vertical,env.num_horizontal),0)
        color_near, color_far = self.conv2d_each_color(dots_kind_matrix)
        all_color = self.conv3d_all_color(color_near)
        
        linear_input_size = torch.flatten(all_color).size(dim=0)
        
        self.layer1 = nn.Linear(linear_input_size, n_squre)
        self.layer2 = nn.Linear(n_squre, n_squre)
        self.layer3 = nn.Linear(n_squre, 1)
        
        self.all_layers = [self.conv_each_color_near, \
                           self.conv_each_color_far, \
                           self.conv_all_color, \
                           self.layer1, self.layer2, self.layer3]
        for ii in self.all_layers:
            nn.init.normal(ii.weight, mean=0,std=0.3)

    def conv2d_each_color(self, dots_kind_matrix):
        for ii in range(self.num_kind):
            # それぞれの色が近場で消せそうか判断する
            each_color_mat = dots_kind_matrix == ii + 1
            each_color_mat = each_color_mat * 1.0 # convert True into 1.0
            each_color_mat = torch.tensor(each_color_mat, dtype=torch.float32, device=device).unsqueeze(0)
            
            each_color_near = self.conv_each_color_near(each_color_mat).unsqueeze(dim=3)
            each_color_far = self.conv_each_color_far(each_color_mat).unsqueeze(dim=3)
            if ii == 0:
                color_near = each_color_near
                color_far = each_color_far
            else:
                # それぞれが消せそうかを3次元的に結合する
                color_near = torch.cat((color_near, each_color_near), dim=3)
                color_far = torch.cat((color_far, each_color_far), dim=3)
                
        return color_near, color_far
    
    def conv3d_all_color(self, color_near):
        linear_input_mat = self.conv_all_color(color_near)
        for ii in range(self.num_kind):
            adding_layer = linear_input_mat[ii,:,:,ii].unsqueeze(2)
            if ii == 0:
                all_color = adding_layer
            else:
                all_color = torch.cat((all_color, adding_layer), dim=2)
        return all_color

    def forward(self, dots_kind_matrix):
        # dots_kind_matrixはnumpy.array
        
        # 現状は近くでつながっている奴だけ考慮
        color_near, color_far = self.conv2d_each_color(dots_kind_matrix)
        all_color = self.conv3d_all_color(color_near)
        x = self.layer1(torch.flatten(all_color))
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x