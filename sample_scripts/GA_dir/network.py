import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from config import elitism_pct, mutation_prob, weights_mutate_power, device
from functions import UI_dots as ui
from functions import engine_dots as eg

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
            nn.init.normal(ii.bias, mean=0,std=0.12)

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
                                           padding = int(full_kernel_size/2), bias=False )
                                           
        self.conv2d_not_each_color = nn.Conv2d(in_channels=1, out_channels=1, \
                                                kernel_size=full_kernel_size, \
                                                padding = int(full_kernel_size/2), bias=False )
                                                
        self.conv2d_not_each_color_2nd = nn.Conv2d(in_channels=1, out_channels=1, \
                                                kernel_size=full_kernel_size, \
                                                padding = int(full_kernel_size/2), bias=False )
                                                
        self.conv2d_empty = nn.Conv2d(in_channels=1, out_channels=1, \
                                        kernel_size=full_kernel_size, \
                                        padding = int(full_kernel_size/2), bias=False )
        
        # ダミーを入れて、変化を把握
        dots_kind_matrix_3D = np.full((env.num_vertical,env.num_horizontal),0)
        color_mat_3d = self.generate_linear_input(dots_kind_matrix_3D)
        
        linear_input_size = color_mat_3d.size(dim=1)
        
        n_squre = 100
        self.layer1 = nn.Linear(linear_input_size, n_squre, bias=False)
        self.layer2 = nn.Linear(n_squre, n_squre, bias=False)
        self.layer3 = nn.Linear(n_squre, 1, bias=False) # 何連鎖できそうなのかの期待値を返すイメージ
        
        self.all_layers = [self.layer1, self.layer2, self.layer3, \
                           self.conv2d_each_color, self.conv2d_not_each_color, self.conv2d_empty,\
                           self.conv2d_not_each_color_2nd]
        for ii in self.all_layers:
            nn.init.normal(ii.weight, mean=0,std=0.3)
            if not(ii.bias is None):
                nn.init.normal(ii.bias, mean=0,std=0.3)

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
        not_each_color_mat_log = torch.zeros(\
                                             size=(\
                                                   dots_kind_matrix_3D.shape[0], \
                                                   dots_kind_matrix_3D.shape[1], \
                                                   dots_kind_matrix_3D.shape[2], \
                                                   dots_kind_matrix_3D.shape[3], \
                                                   self.num_kind \
                                                   ))
        
        # 空の配置に基づいたつながりそうかの評価 
        # 空が周りに多いところはつながりやすい?
        # 想定は empty_value > 0 で, 大きいだけつながりやすい
        # empty は共通だからここで初期化しておく
        empty = (dots_kind_matrix_3D == 0) * 1.0
        empty_value = self.conv2d_empty(empty)
        
        for ii in range(self.num_kind):
            each_color_mat = (dots_kind_matrix_3D == ii + 1) * 1.0
            not_each_color_mat = (each_color_mat + empty == 0) * 1.0
            
            # 各色の配置のみに基づいたつながりそうかの評価
            # 想定は each_color_value > 0 で, 大きいだけつながりやすい
            each_color_value = self.conv2d_each_color(each_color_mat)
            
            each_color_value += empty_value
            
            # 他の色の配置のみに基づいたつながらなさそうかの評価
            # 他の色が多いところは消しづらい?
            # 想定は not_each_color_mat > 0 で, 大きいだけつながりづらい
            not_each_color_mat = self.conv2d_not_each_color(not_each_color_mat)
            each_color_value -= not_each_color_mat
            
            # 着目色以外の座標は無評価
            each_color_value[torch.logical_not(each_color_mat)] = 0
            
            all_color_value = all_color_value + each_color_value
            
            # 使いまわしように保存
            each_color_mat_log[:,:,:,:,ii] = each_color_mat
            each_color_value_log[:,:,:,:,ii] = each_color_value
            not_each_color_mat_log[:,:,:,:,ii] = not_each_color_mat
                
        # all_color_value に基づいて, 改めて消しづらさを考える.
        # 他色であっても, そこが消しやすい場所であるなら, 評価値を下げる必要はない?
        # kernel size を奇数にしたから形が決定されるので, ここで初期化しておく
        all_color_value_2nd = all_color_value
        all_color_value_2nd = 1/(1 + torch.exp(-all_color_value_2nd) )
        
        # all_color_value_2nd を何回更新するか. 
        # 1回ごとに連鎖数が1回増える想定. 
        # 例:   はじめ底の青3つがその上の赤3つのせいで消しづらくて, 赤3つはその上の緑3つのせいで消しづらいって解釈してたけど, 
        #       最初期の all_color_value_2nd によって緑3つが消しやすいって分かって, 
        #       1回目の all_color_value_2nd によって更に赤が消しやすいって分かって,
        #       2回目の all_color_value_2nd によって更に青が消しやすいって分かるイメージ
        # sigmoid で正規化するか悩むところ.
        num_depth = 4
        for reading_depth in range(num_depth):
            # 以前の情報をコピーして, 初期化
            all_color_value_2nd_before = all_color_value_2nd.detach().clone()
            all_color_value_2nd = torch.zeros_like(dots_kind_matrix_3D)
            for ii in range(self.num_kind):
                each_color_mat = each_color_mat_log[:,:,:,:,ii]
                # 先ほどと異なり, 他色の座標の評価は消しやすさに基づく
                not_each_color_mat_2nd = not_each_color_mat_log[:,:,:,:,ii] * all_color_value_2nd_before
                
                # 各色の配置のみに基づいたつながりそうかの評価
                # 想定は each_color_value > 0 で, 大きいだけつながりやすい
                each_color_value = each_color_value_log[:,:,:,:,ii]
                
                # 空の配置に基づいたつながりそうかの評価 
                # 空が周りに多いところはつながりやすい?
                # 想定は empty_value > 0 で, 大きいだけつながりやすい
                each_color_value += empty_value
                
                # 他の色の配置のみに基づいたつながらなさそうかの評価
                # 他の色が多いところは消しづらい?
                # 想定は not_each_color_mat_2nd > 0 で, 大きいだけつながりやすい
                not_each_color_mat_2nd = self.conv2d_not_each_color_2nd(not_each_color_mat_2nd)
                each_color_value += not_each_color_mat_2nd
                
                # 着目色以外の座標は無評価
                each_color_value[torch.logical_not(each_color_mat)] = 0
                
                # all_color_value_2nd に書き加え
                all_color_value_2nd = all_color_value_2nd + each_color_value
                
            all_color_value_2nd = 1/(1 + torch.exp(-all_color_value_2nd) )
        
        
        linear_size = int(all_color_value_2nd.numel()/all_color_value_2nd.shape[0])
        all_color_value_2nd = torch.reshape(all_color_value_2nd, (all_color_value_2nd.shape[0],linear_size))
        return all_color_value_2nd

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # #     
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

class simple_linear_layer(nn.Module):

    def __init__(self, env):
        super(simple_linear_layer, self).__init__()
        
        self.num_kind = env.num_kind
        
        # ダミーを入れて、変化を把握
        dots_kind_matrix_3D = np.full((env.num_vertical,env.num_horizontal),0)
        color_mat_3d = self.generate_linear_input(dots_kind_matrix_3D)
        
        linear_input_size = color_mat_3d.size(dim=1)
        
        n_squre = linear_input_size
        self.layer_in = nn.Linear(linear_input_size, n_squre, bias=False)
        self.layer1 = nn.Linear(n_squre, n_squre, bias=True)
        self.layer2 = nn.Linear(n_squre, n_squre, bias=True)
        self.layer3 = nn.Linear(n_squre, n_squre, bias=True)
        self.layer_out = nn.Linear(n_squre, 1, bias=True) # 何連鎖できそうなのかの期待値を返すイメージ
        
        self.all_layers = [self.layer_in, self.layer_out, \
                           self.layer1, \
                           self.layer2, \
                           self.layer3, \
                           ]
        for ii in self.all_layers:
            nn.init.normal(ii.weight, mean=0,std=0.04)
            if not(ii.bias is None):
                nn.init.normal(ii.bias, mean=0,std=0.03)

    def forward(self, dots_kind_matrix_3D):
        # dots_kind_matrix_3Dはnumpy.array
        
        color_mat_3d = self.generate_linear_input(dots_kind_matrix_3D)
        
        x = self.layer_in(color_mat_3d)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer1(x)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer2(x)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer3(x)
        x = F.leaky_relu(input=x,negative_slope=0.2)
        x = self.layer_out(x)
        x = F.sigmoid(x) * 10
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
        
        # Tensorに変換
        dots_kind_matrix_3D = torch.tensor(dots_kind_matrix_3D, dtype=torch.int32, device=device)
        
        # 各座標の色フラグ. 対等に扱うために0/1に直す.
        each_color_mat_log = torch.zeros(\
                                         size=(\
                                               dots_kind_matrix_3D.shape[0], \
                                               dots_kind_matrix_3D.shape[1], \
                                               dots_kind_matrix_3D.shape[2], \
                                               self.num_kind+1 \
                                               ))
        
        for ii in range(self.num_kind+1):
            each_color_mat = (dots_kind_matrix_3D == ii) * 1.0
            each_color_mat_log[:,:,:,ii] = each_color_mat

        
        linear_size = int(each_color_mat_log.numel()/each_color_mat_log.shape[0])
        each_color_mat_log = torch.reshape(each_color_mat_log, (each_color_mat_log.shape[0],linear_size))
        return each_color_mat_log

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # #     
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

class CNN_symmetry(nn.Module):

    def __init__(self, env, connected_threshold=eg.connected_threshold_default):
        super(CNN_symmetry, self).__init__()
        
        self.num_kind = env.num_kind
        self.connected_threshold=connected_threshold
        
        half_kernel_size = max([env.num_vertical-2,env.num_horizontal])
        half_kernel_size = int(half_kernel_size/2)
        # 奇数辺長の中で最大のものにする. full_kernel_size:3->3, 4->5, 5->5
        full_kernel_size = half_kernel_size*2 + 1
        
        self.half_kernel_size = half_kernel_size
        self.full_kernel_size = full_kernel_size
        
        # list for conv2d whose weight will be symmetric
        self.symmetric_conv2d = []
        
        self.conv2d_each_color = nn.Conv2d(in_channels=1, out_channels=1, \
                                           kernel_size=full_kernel_size, \
                                           padding = half_kernel_size, bias=False )
            
        self.conv2d_not_each_color = nn.Conv2d(in_channels=1, out_channels=1, \
                                                kernel_size=full_kernel_size, \
                                                padding = half_kernel_size, bias=False )
                                                
        self.conv2d_not_each_color_2nd = nn.Conv2d(in_channels=1, out_channels=1, \
                                                kernel_size=full_kernel_size, \
                                                padding = half_kernel_size, bias=False )
                                                
        self.conv2d_empty = nn.Conv2d(in_channels=1, out_channels=1, \
                                        kernel_size=full_kernel_size, \
                                        padding = half_kernel_size, bias=False )
        
        self.symmetric_conv2d = [self.conv2d_each_color, \
                                 self.conv2d_not_each_color, \
                                 self.conv2d_not_each_color_2nd, \
                                 self.conv2d_empty]
        
        
        # make a list of Conv2 and dummy linear layers pair
        self.pair_conv2d_dummy_linear = []
        
        for conv2d_tmp in self.symmetric_conv2d:
            # 正の係数を持つ linear を用意
            linear_tmp = nn.Linear(connected_threshold, full_kernel_size, bias=False)
            linear_tmp.weight.data = torch.abs(linear_tmp.weight.data) * 0.1
            
            # conv2d を0で初期化
            conv2d_tmp.weight.data = torch.zeros_like(conv2d_tmp.weight)
            
            # 真ん中を中心に linear で置換
            conv2d_tmp.weight.data[0,0,:,half_kernel_size:half_kernel_size+connected_threshold]\
                = linear_tmp.weight
            conv2d_tmp.weight.data[0,0,:,half_kernel_size-connected_threshold+1:half_kernel_size+1]\
                = torch.flip(linear_tmp.weight,dims=(1,))
                
            self.pair_conv2d_dummy_linear.append([conv2d_tmp, linear_tmp])
        
        # ダミーを入れて、変化を把握
        dots_kind_matrix_3D = np.full((env.num_vertical,env.num_horizontal),0)
        color_mat_3d = self.generate_linear_input(dots_kind_matrix_3D)
        
        linear_input_size = color_mat_3d.size(dim=1)
        
        n_squre = 100
        self.layer1 = nn.Linear(linear_input_size, n_squre, bias=False)
        self.layer2 = nn.Linear(n_squre, n_squre, bias=True)
        self.layer3 = nn.Linear(n_squre, 1, bias=True) # 何連鎖できそうなのかの期待値を返すイメージ
        
        self.all_layers = [self.layer1, self.layer2, self.layer3]
        # 純粋な Linear 層だけ初期化
        for ii in self.all_layers:
            # nn.init.normal(ii.weight, mean=0,std=0.15) # for num_next_2dots = 1
            nn.init.normal(ii.weight, mean=0,std=0.2) # for num_next_2dots = 2
            if not(ii.bias is None):
                nn.init.normal(ii.bias, mean=0,std=0.1)
        
        # dummpy Linear も加えておく
        for pair_conv2d_dummy_linear in self.pair_conv2d_dummy_linear:
            linear_tmp =pair_conv2d_dummy_linear[1]
            self.all_layers.append(linear_tmp)

    def forward(self, dots_kind_matrix_3D):
        # dots_kind_matrix_3Dはnumpy.array
        
        connected_threshold = self.connected_threshold
        half_kernel_size = self.half_kernel_size
        full_kernel_size = self.full_kernel_size
        
        for pair_conv2d_dummy_linear in self.pair_conv2d_dummy_linear:
            conv2d_tmp =pair_conv2d_dummy_linear[0]
            linear_tmp =pair_conv2d_dummy_linear[1]
            # linear が population などで変更されているはず
            # ただし正に限定しておく
            linear_tmp.weight.data = torch.abs(linear_tmp.weight.data)
            # 真ん中を中心に linear で置換
            conv2d_tmp.weight.data[0,0,:,half_kernel_size:half_kernel_size+connected_threshold]\
                = linear_tmp.weight
            conv2d_tmp.weight.data[0,0,:,half_kernel_size-connected_threshold+1:half_kernel_size+1]\
                = torch.flip(linear_tmp.weight,dims=(1,))
        
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
        not_each_color_mat_log = torch.zeros(\
                                             size=(\
                                                   dots_kind_matrix_3D.shape[0], \
                                                   dots_kind_matrix_3D.shape[1], \
                                                   dots_kind_matrix_3D.shape[2], \
                                                   dots_kind_matrix_3D.shape[3], \
                                                   self.num_kind \
                                                   ))
        
        # 空の配置に基づいたつながりそうかの評価 
        # 空が周りに多いところはつながりやすい?
        # 想定は empty_value > 0 で, 大きいだけつながりやすい
        # empty は共通だからここで初期化しておく
        empty = (dots_kind_matrix_3D == 0) * 1.0
        empty_value = self.conv2d_empty(empty)
        
        for ii in range(self.num_kind):
            each_color_mat = (dots_kind_matrix_3D == ii + 1) * 1.0
            not_each_color_mat = (each_color_mat + empty == 0) * 1.0
            
            # 各色の配置のみに基づいたつながりそうかの評価
            # 想定は each_color_value > 0 で, 大きいだけつながりやすい
            each_color_value = self.conv2d_each_color(each_color_mat)
            
            # 他の色の配置のみに基づいたつながらなさそうかの評価
            # 他の色が多いところは消しづらい?
            # 想定は not_each_color_mat > 0 で, 大きいだけつながりづらい
            not_each_color_mat = self.conv2d_not_each_color(not_each_color_mat)
            
            each_color_value += empty_value
            each_color_value -= not_each_color_mat
            
            # 着目色以外の座標は無評価
            each_color_value[torch.logical_not(each_color_mat)] = 0
            
            all_color_value = all_color_value + each_color_value
            
            # 使いまわしように保存
            each_color_mat_log[:,:,:,:,ii] = each_color_mat
            each_color_value_log[:,:,:,:,ii] = each_color_value
            not_each_color_mat_log[:,:,:,:,ii] = not_each_color_mat
                
        # all_color_value に基づいて, 改めて消しづらさを考える.
        # 他色であっても, そこが消しやすい場所であるなら, 評価値を下げる必要はない?
        # kernel size を奇数にしたから形が決定されるので, ここで初期化しておく
        # all_color_value は正も負も動くから sigmoidでの初期化は素直でよい
        all_color_value_2nd = all_color_value
        all_color_value_2nd = 1/(1 + torch.exp(-all_color_value_2nd) )
        
        # all_color_value_2nd を何回更新するか. 
        # 1回ごとに連鎖数が1回増える想定. 
        # 例:   はじめ底の青3つがその上の赤3つのせいで消しづらくて, 赤3つはその上の緑3つのせいで消しづらいって解釈してたけど, 
        #       最初期の all_color_value_2nd によって緑3つが消しやすいって分かって, 
        #       1回目の all_color_value_2nd によって更に赤が消しやすいって分かって,
        #       2回目の all_color_value_2nd によって更に青が消しやすいって分かるイメージ
        # sigmoid で正規化するか悩むところ.
        num_depth = 2
        for reading_depth in range(num_depth):
            if dots_kind_matrix_3D.shape[0] > 1:
                print("", end="")
            
            # 以前の情報をコピーして, 初期化
            all_color_value_2nd_before = all_color_value_2nd.detach().clone()
            all_color_value_2nd = torch.zeros_like(dots_kind_matrix_3D)
            for ii in range(self.num_kind):
                each_color_mat = each_color_mat_log[:,:,:,:,ii]
                # 先ほどと異なり, 他色の座標の評価は消しやすさに基づく
                not_each_color_mat_2nd = not_each_color_mat_log[:,:,:,:,ii] * all_color_value_2nd_before
                
                # 各色の配置のみに基づいたつながりそうかの評価
                # 想定は each_color_value > 0 で, 大きいだけつながりやすい
                each_color_value = each_color_value_log[:,:,:,:,ii]
                
                # 他の色の配置のみに基づいたつながらなさそうかの評価
                # 他の色が多いところは消しづらい?
                # 想定は not_each_color_mat_2nd > 0 で, 大きいだけつながりやすい
                not_each_color_mat_2nd = self.conv2d_not_each_color_2nd(not_each_color_mat_2nd)
                
                each_color_value += empty_value
                each_color_value += not_each_color_mat_2nd
                
                # 着目色以外の座標は無評価
                each_color_value[torch.logical_not(each_color_mat)] = 0
                
                # all_color_value_2nd に書き加え
                all_color_value_2nd = all_color_value_2nd + each_color_value
                
                if dots_kind_matrix_3D.shape[0] > 1:
                    print("", end="")
            
            if dots_kind_matrix_3D.shape[0] > 1:
                print("", end="")
            # all_color_value_2nd は each_color_value > 0, empty > 0, not_each_color_mat_2nd > 0 なので
            # all_color_value_2nd > 0 だから素直なsigmoid だと 0.5~1 になってしまう.
            # だから 2倍して1引く
            all_color_value_2nd = 1/(1+torch.exp(-all_color_value_2nd) ) * 2 - 1
        
        if dots_kind_matrix_3D.shape[0] > 1:
            print("", end="")
        linear_size = int(all_color_value_2nd.numel()/all_color_value_2nd.shape[0])
        all_color_value_2nd = torch.reshape(all_color_value_2nd, (all_color_value_2nd.shape[0],linear_size))
        return all_color_value_2nd