import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from config import elitism_pct, mutation_prob, weights_mutate_power, device

class NN_direct_LN_exp(nn.Module):

    def __init__(self, env):
        super(NN_direct_LN_exp, self).__init__()
        
        self.num_kind = env.num_kind
        
        full_kernel_size = max([env.num_vertical-2,env.num_horizontal])
        
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
        dots_kind_matrix_3D = np.full((env.num_vertical,env.num_horizontal),0)
        color_mat_3d = self.generate_linear_input(dots_kind_matrix_3D)
        
        linear_input_size = color_mat_3d.size(dim=1)
        
        n_squre = 200
        self.layer1 = nn.Linear(linear_input_size, n_squre)
        self.layer2 = nn.Linear(n_squre, n_squre)
        self.layer3 = nn.Linear(n_squre, 1) # 何連鎖できそうなのかの期待値を返すイメージ
        
        self.all_layers = [self.layer1, self.layer2, self.layer3, \
                           self.conv2d_each_color, self.conv2d_not_each_color, self.conv2d_empty]
        for ii in self.all_layers:
            nn.init.normal(ii.weight, mean=0,std=0.12)

    def forward(self, dots_kind_matrix_3D):
        # dots_kind_matrix_3Dはnumpy.array
        
        color_mat_3d = self.generate_linear_input(dots_kind_matrix_3D)
        
        x = self.layer1(color_mat_3d)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer2(x)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer3(x)
        return x
    
    def generate_linear_input(self, dots_kind_matrix_3D):
        # trimming
        if dots_kind_matrix_3D.ndim > 2:
            dots_kind_matrix_3D = dots_kind_matrix_3D[0:-2,:,:]
        else:
            # 空次元を挿入しておく
            dots_kind_matrix_3D = dots_kind_matrix_3D[0:-2,:,np.newaxis]
        
        # 3次元目を一番前にもってくる. 入力が複数になったように Conv2D には見せかける
        dots_kind_matrix_3D = dots_kind_matrix_3D.transpose(2,0,1)
        
        # Conv2D の inputchannelに合わせて空次元を挿入.
        dots_kind_matrix_3D = dots_kind_matrix_3D[:,np.newaxis,:,:]
        
        # TODO: Tensorの1次元目を増やすことで、forループを回避できないか検討
        for ii in range(self.num_kind):
            each_color_mat = (dots_kind_matrix_3D == ii + 1) * 1.0
            empty = (dots_kind_matrix_3D == 0) * 1.0
            not_each_color_mat = np.logical_not(each_color_mat + empty > 0) * 1.0
            
            each_color_mat = torch.tensor(each_color_mat, dtype=torch.float32, device=device)
            each_color_mat = self.conv2d_each_color(each_color_mat) # つながりそうかの評価
            
            empty = torch.tensor(empty, dtype=torch.float32, device=device)
            empty = self.conv2d_empty(empty) # つながりそうかの評価 # 空のところはつながりやすい？
            
            each_color_mat += empty
            
            not_each_color_mat = torch.tensor(not_each_color_mat, dtype=torch.float32, device=device)
            not_each_color_mat = self.conv2d_not_each_color(not_each_color_mat) # つながらなさそうかの評価
            
            each_color_mat -= not_each_color_mat
            
            each_color_mat = each_color_mat.unsqueeze(4) # 5次元化
            
            if ii == 0:
                color_mat_3d = each_color_mat # I don't know empty initialization of tensor, so initialize at the first loop
            else:
                color_mat_3d = torch.cat((color_mat_3d, each_color_mat), dim=4)
        linear_size = int(color_mat_3d.numel()/color_mat_3d.shape[0])
        color_mat_3d = torch.reshape(color_mat_3d, (color_mat_3d.shape[0],linear_size))
        return color_mat_3d
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # #     
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

class NN_each_LN_exp(nn.Module):

    def __init__(self, env):
        super(NN_each_LN_exp, self).__init__()
        
        self.num_kind = env.num_kind
        
        full_kernel_size = max([env.num_vertical-2,env.num_horizontal])
        # 奇数辺長の中で最大のものにする. full_kernel_size:3->3, 4->5, 5->5
        full_kernel_size = int(full_kernel_size/2)*2 + 1
        
        self.conv2d_each_color = nn.Conv2d(in_channels=1, out_channels=1, \
                                           kernel_size=full_kernel_size, \
                                           padding = int(full_kernel_size/2) )
                                           
        self.conv2d_not_each_color = nn.Conv2d(in_channels=1, out_channels=1, \
                                                kernel_size=full_kernel_size, \
                                                padding = int(full_kernel_size/2) )
                                                
        self.conv2d_not_each_color_2nd = nn.Conv2d(in_channels=1, out_channels=1, \
                                                kernel_size=full_kernel_size, \
                                                padding = int(full_kernel_size/2) )
                                                
        self.conv2d_empty = nn.Conv2d(in_channels=1, out_channels=1, \
                                        kernel_size=full_kernel_size, \
                                        padding = int(full_kernel_size/2) )
        
        # 各座標に対してそれぞれ1連鎖目で消える期待値を返すイメージ
        self.conv2d_LN_1_exp = nn.Conv2d(in_channels=1, out_channels=1, \
                                        kernel_size=full_kernel_size, \
                                        padding = int(full_kernel_size/2) )
            
        # 各座標に対してそれぞれ2連鎖目で消える期待値を返すイメージ
        self.conv2d_LN_2_exp = nn.Conv2d(in_channels=1, out_channels=1, \
                                        kernel_size=full_kernel_size, \
                                        padding = int(full_kernel_size/2) )
        
        # ダミーを入れて、変化を把握
        dots_kind_matrix_3D = np.full((env.num_vertical,env.num_horizontal),0)
        color_mat_3d = self.generate_linear_input(dots_kind_matrix_3D)
        
        linear_input_size = color_mat_3d.size(dim=1)
        
        n_squre = 100
        self.layer1 = nn.Linear(linear_input_size, n_squre)
        self.layer2 = nn.Linear(n_squre, n_squre)
        self.layer3 = nn.Linear(n_squre, 1) # 何連鎖できそうなのかの期待値を返すイメージ
        
        self.all_layers = [self.layer1, self.layer2, self.layer3, \
                           self.conv2d_each_color, self.conv2d_not_each_color, self.conv2d_empty,\
                           self.conv2d_not_each_color_2nd]
        for ii in self.all_layers:
            nn.init.normal(ii.weight, mean=0,std=0.14)

    def forward(self, dots_kind_matrix_3D):
        # dots_kind_matrix_3Dはnumpy.array
        
        color_mat_3d = self.generate_linear_input(dots_kind_matrix_3D)
        
        x = self.layer1(color_mat_3d)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer2(x)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer3(x)
        return x
    
    def generate_linear_input(self, dots_kind_matrix_3D):
        # trimming
        if dots_kind_matrix_3D.ndim > 2:
            dots_kind_matrix_3D = dots_kind_matrix_3D[0:-2,:,:]
        else:
            # 空次元を挿入しておく
            dots_kind_matrix_3D = dots_kind_matrix_3D[0:-2,:,np.newaxis]
        
        # 3次元目を一番前にもってくる. 入力が複数になったように Conv2D には見せかける
        dots_kind_matrix_3D = dots_kind_matrix_3D.transpose(2,0,1)
        
        # Conv2D の inputchannelに合わせて空次元を挿入.
        dots_kind_matrix_3D = dots_kind_matrix_3D[:,np.newaxis,:,:]
        
        # Tensorに変換
        dots_kind_matrix_3D = torch.tensor(dots_kind_matrix_3D, dtype=torch.float32, device=device)
        
        # ある色に対して, その色か, 空か, それ以外の色か, に基づいて消しやすさを導く.
        # 想定は all_color_value > 0 で, 大きいだけつながりやすい
        # kernel size を奇数にしたから形が決定されるので, ここで初期化しておく
        all_color_value = torch.zeros_like(dots_kind_matrix_3D)
        
        # のちに定義する each_color_value の保存用.
        each_color_value_log = torch.zeros(\
                                           size=(\
                                                 dots_kind_matrix_3D.shape[0], \
                                                 dots_kind_matrix_3D.shape[1], \
                                                 dots_kind_matrix_3D.shape[2], \
                                                 dots_kind_matrix_3D.shape[3], \
                                                 self.num_kind \
                                                 ))
        each_color_mat_log = torch.zeros(\
                                         size=(\
                                               dots_kind_matrix_3D.shape[0], \
                                               dots_kind_matrix_3D.shape[1], \
                                               dots_kind_matrix_3D.shape[2], \
                                               dots_kind_matrix_3D.shape[3], \
                                               self.num_kind \
                                               ))
        
        # 空の配置に基づいたつながりそうかの評価 
        # 空が周りに多いところはつながりやすい?
        # 想定は empty > 0 で, 大きいだけつながりやすい
        # empty は共通だからここで初期化しておく
        empty = (dots_kind_matrix_3D == 0) * 1.0
        empty = self.conv2d_empty(empty)
        
        for ii in range(self.num_kind):
            each_color_mat = (dots_kind_matrix_3D == ii + 1) * 1.0
            not_each_color_mat = torch.logical_not(each_color_mat + empty > 0) * 1.0
            
            # 各色の配置のみに基づいたつながりそうかの評価
            # 想定は each_color_value > 0 で, 大きいだけつながりやすい
            each_color_value = self.conv2d_each_color(each_color_mat)
            
            each_color_mat_log[:,:,:,:,ii] = each_color_mat
            each_color_value_log[:,:,:,:,ii] = each_color_value
            
            each_color_value += empty
            
            # 他の色の配置のみに基づいたつながらなさそうかの評価
            # 他の色が多いところは消しづらい?
            # 想定は not_each_color_mat > 0 で, 大きいだけつながりづらい
            not_each_color_mat = self.conv2d_not_each_color(not_each_color_mat)
            each_color_value -= not_each_color_mat
            
            # 着目色以外の座標は無評価
            each_color_value[torch.logical_not(each_color_mat)] = 0
            
            all_color_value = all_color_value + each_color_value
                
        # all_color_value に基づいて, 改めて消しづらさを考える.
        # 他色であっても, そこが消しやすい場所であるなら, 評価値を下げる必要はない?
        # kernel size を奇数にしたから形が決定されるので, ここで初期化しておく
        all_color_value_2nd = all_color_value
        
        # all_color_value_2nd を何回更新するか. 
        # 1回ごとに連鎖数が1回増える想定. 
        # 例:   はじめ底の青3つがその上の赤3つのせいで消しづらくて, 赤3つはその上の緑3つのせいで消しづらいって解釈してたけど, 
        #       最初期の all_color_value_2nd によって緑3つが消しやすいって分かって, 
        #       1回目の all_color_value_2nd によって更に赤が消しやすいって分かって,
        #       2回目の all_color_value_2nd によって更に青が消しやすいって分かるイメージ
        # sigmoid で正規化するか悩むところ.
        num_depth = 1
        for reading_depth in range(num_depth):
            for ii in range(self.num_kind):
                each_color_mat = each_color_mat_log[:,:,:,:,ii]
                # 先ほどと異なり, 他色の座標の評価は消しやすさに基づく
                not_each_color_mat_2nd = (torch.logical_not(each_color_mat + empty > 0) * 1.0) * all_color_value_2nd
                
                # 各色の配置のみに基づいたつながりそうかの評価
                # 想定は each_color_value > 0 で, 大きいだけつながりやすい
                each_color_value = each_color_value_log[:,:,:,:,ii]
                
                # 空の配置に基づいたつながりそうかの評価 
                # 空が周りに多いところはつながりやすい?
                # 想定は empty > 0 で, 大きいだけつながりやすい
                each_color_value += empty
                
                # 他の色の配置のみに基づいたつながらなさそうかの評価
                # 他の色が多いところは消しづらい?
                # 想定は not_each_color_mat_2nd > 0 で, 大きいだけつながりやすい
                not_each_color_mat_2nd = self.conv2d_not_each_color_2nd(not_each_color_mat_2nd)
                each_color_value += not_each_color_mat_2nd
                
                # 着目色以外の座標は無評価
                each_color_value[torch.logical_not(each_color_mat)] = 0
                
                all_color_value_2nd = all_color_value_2nd + each_color_value
        
        
        linear_size = int(all_color_value_2nd.numel()/all_color_value_2nd.shape[0])
        all_color_value_2nd = torch.reshape(all_color_value_2nd, (all_color_value_2nd.shape[0],linear_size))
        return all_color_value_2nd
