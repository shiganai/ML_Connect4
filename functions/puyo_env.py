
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import UI_dots as ui
from functions import engine_dots as eg

import torch
device = 'cpu'

class puyo_env:
    
    # def __init__(self):
                
    def __init__(self, \
            dots_kind_matrix=None, \
            num_horizontal = eg.num_horizontal_default, \
            num_vertical = eg.num_vertical_default, \
            num_kind = eg.num_kind_default, \
            num_dummy_kind = eg.num_dummy_kind_default, \
            num_next_2dots = 3):
            
        print("construct of puyo_env is called")

        self.num_horizontal = num_horizontal
        self.num_vertical = num_vertical
        self.num_kind = num_kind
        self.num_dummy_kind = num_dummy_kind
        self.num_next_2dots = num_next_2dots
        
        self.num_candidate = self.num_horizontal * 2 + (self.num_horizontal-1) * 2
        self.num_dots = self.num_horizontal * self.num_vertical
        
        self.action_space = self.action_space(self)
        
        self.turn_count_threshold = -1 + 4 * 5
    
# =============================================================================
#     copying openaigym sample
# =============================================================================
    def reset(self):
        # print("reset of puyo_env is called")
        self.reset_kind_matrix()
        self.generate_next_2dots()
        
        self.turn_count = 0 # temporary count for stop at some count
        
        state = self.update_state()
        self.list_candidate()
        info = None # No idea what info is
        
        return state, info
        
    def step(self, action):
        self.dots_kind_matrix = self.candidates[action]
        loop_num = self.drop_candidate()
        
        ########## refresh dots
        self.update_next_2dots()
        self.list_candidate()
        
        ########## define returning value
        state = self.update_state()
        observation = state
        reward = float(loop_num**2)
        
        # if loop_num < 2:
        #     reward = 0
        
        self.turn_count = self.turn_count + 1
            
        terminated = (self.turn_count > self.turn_count_threshold) or (self.get_terminated())
        # terminated = self.get_terminated()
        
        truncated = False
        
        info = None # info is dummpy return value
        
        
        return observation, reward, terminated, truncated, info
        
    class action_space():
        def __init__(self, master):
            print("construct of action_space is called")
            self.master = master
            self.n = self.master.num_candidate
            
        def sample(self):
            return np.random.randint(0,self.n)
        
# =============================================================================
#     original
# =============================================================================
        
    def reset_kind_matrix(self):
        self.dots_kind_matrix = np.full( (self.num_vertical, self.num_horizontal), 0 )
        
    def generate_next_2dots(self):
        self.next_2dots = np.random.randint(1,self.num_kind+1,(2,self.num_next_2dots))
    
    def update_next_2dots(self):
        adding_2dots = np.random.randint(1,self.num_kind+1,(1,2))
        next_2dots = []
        for ii in range(1,self.num_next_2dots):
            next_2dots.append(self.next_2dots[:,ii])
        next_2dots.append(adding_2dots)
        
        self.next_2dots = np.transpose(np.vstack(next_2dots))
        
    def update_state(self):
        dots_kind_matrix = np.copy(self.dots_kind_matrix)
        
        for horizontal_index in range(self.num_next_2dots):
            dots_kind_matrix[-2,horizontal_index] = self.next_2dots[0,horizontal_index]
            dots_kind_matrix[-1,horizontal_index] = self.next_2dots[1,horizontal_index]
        
        self.dots_kind_matrix_with_candidate = dots_kind_matrix
        
        return dots_kind_matrix.flatten()
    
    def list_candidate(self):
        self.candidates = eg.get_candidate(\
                              dots_kind_matrix=self.dots_kind_matrix, \
                              next_2dots=self.next_2dots[:,0], \
                              listup_same=True)
        if not(len(self.candidates) == self.num_candidate):
            raise Exception("not(len(self.candidates) == self.num_candidate)")
            
        self.candidates_dots_result = []
        self.candidates_loop_num = []
        for candidate in self.candidates:
            dots_transition, loop_num = eg.delete_and_fall_dots_to_the_end(candidate, 4)
            self.candidates_dots_result.append(dots_transition[-1])
            self.candidates_loop_num.append(loop_num)
        
        
    def drop_candidate(self):
        dots_transition, loop_num = eg.delete_and_fall_dots_to_the_end(self.dots_kind_matrix, 4)
        
        self.dots_kind_matrix = dots_transition[-1]
        
        return loop_num
    
    def get_terminated(self):
        terminated = True
        if np.all(self.dots_kind_matrix[-2,:]==0):
            terminated = False
        
        return terminated
    
    def play_one_game(self, model=None, if_disp=False):
        self.reset()
        sum_reward = 0.1
        max_reward = 0
        step_count = 0
        
        dots_transition = []
        
        while True:
            NN_values = []
            exp_values = [] # 期待値評価
            exp_loop_num_values = [] # 連鎖数期待値
            exp_NN_values = [] # NN_value期待値
            for evaled_candidate in self.candidates_dots_result:
                # それぞれの場所に確定している次の2ドットを落とした場合の結果に対して...
                # さらに次の未確定の2ドットに対して連鎖数の期待値と、NN_valueの期待値を得る
                # 連鎖数の期待値は数学的に確定された期待値
                # NN_valueの期待値はNNが予想する更に先の未確定の2ドットを含めた連鎖数の期待値
                loop_num_exp, NN_value_exp = self.list_expectation(evaled_candidate, model)
                exp_loop_num_values.append(loop_num_exp)
                exp_NN_values.append(NN_value_exp)
                
            
            
            # 確定連鎖数が最も大きいものを特定
            best_loop_num_value = np.array(self.candidates_loop_num).max()
            best_loop_num_index = np.where(self.candidates_loop_num == best_loop_num_value)[0]
            if len(best_loop_num_index) > 1:
                best_loop_num_index_best_exp_index = np.array(exp_loop_num_values)[best_loop_num_index].argmax()
                best_loop_num_index = best_loop_num_index[best_loop_num_index_best_exp_index]
            else:
                best_loop_num_index = best_loop_num_index[0]
            
            # 連鎖数期待値評価が最も大きいものを特定
            exp_loop_num_values = np.array(exp_loop_num_values)
            best_loop_num_exp_value = exp_loop_num_values.max()
            best_loop_num_exp_index = np.where(exp_loop_num_values == best_loop_num_exp_value)[0]
            if len(best_loop_num_exp_index) > 1:
                best_loop_num_exp_index = best_loop_num_exp_index[np.random.randint(0, len(best_loop_num_exp_index))]
            else:
                best_loop_num_exp_index = best_loop_num_exp_index[0]
            
            
            # NN_value期待値評価が最も大きいものを特定
            exp_NN_values = np.array(exp_NN_values)
            best_NN_exp_value = exp_NN_values.max()
            best_NN_exp_index = np.where(exp_NN_values == best_NN_exp_value)[0]
            if len(best_NN_exp_index) > 1:
                best_NN_exp_index = best_NN_exp_index[np.random.randint(0, len(best_NN_exp_index))]
            else:
                best_NN_exp_index = best_NN_exp_index[0]
            
            # 選ぶ用の配列作成
            best_values = np.array([best_loop_num_value, best_loop_num_exp_value, best_NN_exp_value])
            best_values_indices = [best_loop_num_index, best_loop_num_exp_index, best_NN_exp_index]
            best_values_labels = ["LN_d", "LN_e", "NN_e"]
            
            choise = best_values.argmax()
            best_index = best_values_indices[choise]
            if if_disp:
                print("LN_d: {}, LN_e: {:>5.2f}, NN_e: {:>5.2f}, ".format(\
                                                              best_loop_num_value, \
                                                              best_loop_num_exp_value, \
                                                              best_NN_exp_value\
                                                              ), end="")
                print("chose " + best_values_labels[choise], end="")
                print(" at turn: " + str(self.turn_count))
                dots_transition.append(self.candidates[best_index])
                
            observation, reward, terminated, truncated, info = self.step(best_index)
                
            sum_reward += reward
            step_count += 1
            
            if reward > max_reward:
                max_reward = reward
            
            if terminated:
                break
        
        # return max_reward + sum_reward, dots_transition
        # return max_reward, dots_transition
        return sum_reward, dots_transition
    
    def list_expectation(self, dots_kind_matrix, model):
        best_loop_num_list = [] # ありえる次の2ドットそれぞれの最高連鎖数
        best_NN_value_list = [] # ありえる次の2ドットそれぞれの最高NN値
        # 色が同じ時も、違う時も、出る確率は同じ。
        # 赤赤が出る確率は1/4*1/4, 赤青が出る確率も1/4*1/4
        
        # next_2dotの組み合わせを全て列挙
        next_2dots_list = []
        
# =============================================================================
#         # 2つの色をどちらも考える場合
#         for ii in range(self.num_kind):
#             for jj in range(ii, self.num_kind):
#                 next_2dots_list.append(np.array([ii+1,jj+1]))
# =============================================================================
        # 1色だけ考える
        for ii in range(self.num_kind):
                next_2dots_list.append(np.array([ii+1,0]))
                
        for next_2dots in next_2dots_list:
            # 赤赤の場合は並べ方は片方だけでいい。確率は計算せず、最高のものを選ぶだけだから。
            dots_exp = eg.get_candidate(\
                            dots_kind_matrix=dots_kind_matrix, \
                            next_2dots=next_2dots, \
                            listup_same=False)
            
            best_loop_num_given2dots = np.NINF
            best_NN_value_given2dots = np.NINF
            for candidate in dots_exp: # 与えられた2ドットを、ある場所においた場合
                # その時のドットの推移と連鎖数
                dots_transition, loop_num_given2dots_givenPlace = eg.delete_and_fall_dots_to_the_end(candidate, 4)
                if not(np.all(dots_transition[-1][-2,:]==0)):
                    # ゲームオーバーなら更新の考慮する必要なし
                    continue
                
                NN_value_given2dots_givenPlace = model(dots_transition[-1])
                NN_value_given2dots_givenPlace = NN_value_given2dots_givenPlace.to('cpu').detach().numpy().copy()
                NN_value_given2dots_givenPlace = NN_value_given2dots_givenPlace[0]
                
                if best_loop_num_given2dots < loop_num_given2dots_givenPlace:
                    best_loop_num_given2dots = loop_num_given2dots_givenPlace
                    
                if best_NN_value_given2dots < NN_value_given2dots_givenPlace:
                    best_NN_value_given2dots = NN_value_given2dots_givenPlace
            
            if best_loop_num_given2dots == np.NINF:
                # もし一度もbest_loop_num_given2dotsが更新されなかった時
                # (ある与えられた2ドットをどこにおいてもゲームオーバーになるとき)(ゲームオーバーになる2ドットが存在するとき)
                best_loop_num_given2dots = -3 # リスクの大きさ。無根拠
                best_NN_value_given2dots = -3 # リスクの大きさ。無根拠
                
            best_loop_num_list.append(best_loop_num_given2dots)
            best_NN_value_list.append(best_NN_value_given2dots)
                
        loop_num_exp = np.array(best_loop_num_list).mean()
        NN_value_exp = np.array(best_NN_value_list).mean()
        
        return loop_num_exp, NN_value_exp