
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
        
        self.turn_count_threshold = -1 + 6 * 6
    
# =============================================================================
#     copying openaigym class
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
        self.dots_kind_matrix = self.candidates[:,:,action]
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
            
        
# =============================================================================
#         connected_dots_list, _ = eg.connect_dots(self.dots_kind_matrix)
#         self.connected_dots_list = connected_dots_list
#         for connected_dots in connected_dots_list:
#             if len(connected_dots[0]) > 1:
#                 reward = reward + 1.0/2/2/2
#             if len(connected_dots[0]) > 2:
#                 reward = reward + 1.0/2/2/2
# =============================================================================
        
        self.turn_count = self.turn_count + 1
# =============================================================================
#         terminated = self.get_terminated()
#         if terminated:
#             reward -= 10
# =============================================================================
            
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
        self.candidates = eg.get_candidate_3D(\
                              dots_kind_matrix=self.dots_kind_matrix, \
                              next_2dots=self.next_2dots[:,0], \
                              if_list_all=True)
        if not(self.candidates.shape[2] == self.num_candidate):
            raise Exception("not(len(self.candidates) == self.num_candidate)")
        
        dots_kind_matrix_3D_result, loop_num, dots_transition = \
            eg.delete_and_fall_dots_to_the_end(self.candidates, if_return_only_result=True)
        self.candidates_dots_result = dots_kind_matrix_3D_result
        self.candidates_loop_num = loop_num
        
        
    def drop_candidate(self):
        dots_kind_matrix_3D_result, loop_num, dots_transition = \
            eg.delete_and_fall_dots_to_the_end(self.dots_kind_matrix, if_return_only_result=True)
        
        self.dots_kind_matrix = dots_kind_matrix_3D_result[:,:,0]
        
        return loop_num[0]
    
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
        
        dots_transition_only_result = None
        # dots_transition_3D_list[0]でアクセスできるように初期化
        dots_transition_3D_list = [[]]
        # ラベル付けように追加
        title_for_dots_transition_3D_list = []
        
        while True:
            NN_values = []
            for evaled_index in range(self.candidates_dots_result.shape[2]):
                evaled_candidate = self.candidates_dots_result[:,:,evaled_index]
                if np.all(evaled_candidate[-2,:]==0):
                    sum_value = model(evaled_candidate)
                    sum_value = sum_value.to('cpu').detach().numpy().copy()
                    sum_value = sum_value[0]
                else:
                    sum_value = np.NINF
# =============================================================================
#                 sum_value = 0
#                 for ii in range(self.num_kind):
#                     each_dot_mat = evaled_candidate == ii+1
#                     each_dot_mat = each_dot_mat * 2.0
#                     each_dot_mat[each_dot_mat==0] = 1.0
#                     each_dot_mat[evaled_candidate==0] = 0.0
#                     state = torch.tensor(each_dot_mat, dtype=torch.float32, device=device).unsqueeze(0)
#                     current_value = model(state)
#                     current_value = current_value.to('cpu').detach().numpy().copy()
#                     sum_value += current_value
# =============================================================================
                NN_values.append(sum_value)
                
            
            NN_values = np.array(NN_values)
            # NNの計算値が最も大きいものを特定
            best_NN_value = NN_values.max()
            best_NN_index = np.where(NN_values == best_NN_value)[0]
            if len(best_NN_index) > 1:
                best_NN_index = best_NN_index[np.random.randint(0, len(best_NN_index))]
            else:
                best_NN_index = best_NN_index[0]
            
            best_index = best_NN_index # デフォルトで NN_value の判断を採用
            is_NN_value_chosen = True
            
            # 想定される連鎖数が最も大きいものを特定
            best_loop_num_value = np.array(self.candidates_loop_num).max()
            best_loop_num_index = np.where(self.candidates_loop_num == best_loop_num_value)[0]
            
            is_best_loop_num_index_chosen_randomly = False
            if len(best_loop_num_index) > 1:
                # たまにNN_valueまで一緒の時がある
                NN_values_at_bestLN = NN_values[best_loop_num_index]
                NN_values_max_value_at_bestLN = NN_values_at_bestLN.max()
                # best_loop_num_index 内での NN_value が最大値の位置を把握
                best_loop_num_index_best_NN_index = np.where(NN_values_at_bestLN == NN_values_max_value_at_bestLN)[0]
                if len(best_loop_num_index_best_NN_index) > 1: # NN_valueまで一緒の場合
                    # ランダムに選ぶ
                    best_loop_num_index_best_NN_index = \
                        best_loop_num_index_best_NN_index[np.random.randint(0, len(best_loop_num_index_best_NN_index))]
                    
                    is_best_loop_num_index_chosen_randomly = True
                else:
                    best_loop_num_index_best_NN_index = best_loop_num_index_best_NN_index[0]
                
                # 最後に best_loop_num_index の中から best_loop_num_index_best_NN_index の位置を取得する
                best_loop_num_index = best_loop_num_index[best_loop_num_index_best_NN_index]
                
            else:
                best_loop_num_index = best_loop_num_index[0]
                
            if if_disp:
                print("At turn: {:>3}, best_loop_num: {:>3}, best_NN_value: {:>5.2f}"\
                      .format(self.turn_count, best_loop_num_value, best_NN_value), \
                      end="")
                    
            if best_loop_num_value < 1: # 連鎖がない場合
                chosen_loop_num = self.candidates_loop_num[best_index]
            else: # 連鎖がある場合は NN_value と秤にかける
                if best_loop_num_index == best_NN_index: # 2つの選択結果が同じだった場合
                    is_NN_value_chosen = False
                    if if_disp:
                        print(", and same candidate", end="")
                else:
                    if best_loop_num_value > 9: # 10連鎖以上だったら NN_value 関係なく打つ
                        best_index = best_loop_num_index
                        is_NN_value_chosen = False
                        if if_disp:
                            print(", so chose loop_num", end="")
                    else:
                        if best_loop_num_value > best_NN_value: # 確定連鎖数のほうが大きかったらもう打つ
                            best_index = best_loop_num_index
                            is_NN_value_chosen = False
                            if if_disp:
                                print(", so chose loop_num", end="")
                        else: # NN_valueの期待値が大きいなら次の2ドットに期待. デフォルトで NN_valueを選んでいるから結果の表示以外何もしない
                            if if_disp:
                                print(", so chose NN_value", end="")
                
                chosen_loop_num = self.candidates_loop_num[best_index]
                
                if if_disp:
                    print(", chosen loop_num was {}".format(self.candidates_loop_num[best_index]), end="")
                
                if (self.candidates_loop_num[best_index] == best_loop_num_value) \
                    and (is_NN_value_chosen): # 起こす連鎖数が同じなのに NN_value が選ばれた場合
                    if is_best_loop_num_index_chosen_randomly:
                        # 最高確定連鎖数 と 最高NN_value を持つケースが2つ以上存在して、
                        # 最高確定連鎖数 のインデックスがランダムに選ばれて、
                        # 最高NN_value のインデックスもランダムに選ばれて、
                        # それが一致しない場合
                        if if_disp:
                            print()
                            print("best_loop_num_index was chosen randomly because both loop_num and NN_value are same", \
                                  end="")
                    else:
                        # なぜ起きるか不明. 要デバック状況
                        raise Exception('Undefined case')
            
            if if_disp:
                print()
                
            
                _, _, dots_transition_current_turn = \
                    eg.delete_and_fall_dots_to_the_end(\
                                                       self.candidates[:,:,best_index], \
                                                       if_return_only_result=False)
                        
                # 3Dで入力したとき用に dots_transition_current_turn は [?x?x?, ?x?x?] の形で帰ってくるから、
                # 最初の1つ目を取得. というか len(dots_transition_current_turn) = 1 のはず.
                dots_transition_current_turn = dots_transition_current_turn[0]
                        
                if dots_transition_only_result is None:
                    dots_transition_only_result = self.candidates[:,:,best_index,np.newaxis]
                else:
                    dots_transition_only_result = np.concatenate([\
                                                      dots_transition_only_result, \
                                                      self.candidates[:,:,best_index,np.newaxis],\
                                                      ], axis=2)
                
                if chosen_loop_num < 1: # 連鎖がない場合はdots_transition_3D_listの最後の要素の最後に追加する
                    # プロット用タイトルにはターン数だけ入力
                    title_for_dots_transition_3D_list.append("turn: {}".format(self.turn_count))
                    if dots_transition_3D_list[-1] == []: # まずは初期化
                        dots_transition_3D_list[-1] = self.candidates[:,:,best_index, np.newaxis]
                    else:
                        # もし十分連鎖しない盤面が長くなった場合は改行する.
                        if dots_transition_3D_list[-1].shape[2] > 6:
                            dots_transition_3D_list.append(self.candidates[:,:,best_index, np.newaxis])
                        else:
                            dots_transition_3D_list[-1] = \
                                np.concatenate([\
                                                dots_transition_3D_list[-1],\
                                                self.candidates[:,:,best_index, np.newaxis],\
                                                ],axis=2)
                else: # 連鎖がある場合
                    # タイトルは 初めの2つに turn: ?, chosen loop_num: ? を表示させた後、
                    # 空白を繰り返す.
                    title_for_dots_transition_current_turn = [\
                                                              "turn: {}".format(self.turn_count), \
                                                              "chosen loop_num: {}".format(chosen_loop_num),\
                                                              ]
                    title_for_dots_transition_current_turn.extend([""]*(dots_transition_current_turn.shape[2]-2))
                    title_for_dots_transition_3D_list.extend(title_for_dots_transition_current_turn)
                    if dots_transition_3D_list[-1] == []:
                        # すでに初期化されていた場合. 2回連続で連鎖すると起きる
                        dots_transition_3D_list[-1] = dots_transition_current_turn
                    else:
                        dots_transition_3D_list.append(dots_transition_current_turn)
                        # 次連鎖しない場合用に空を挿入しておく
                        dots_transition_3D_list.append([])
                    
            observation, reward, terminated, truncated, info = self.step(best_index)
                
            sum_reward += reward
            step_count += 1
            
            if reward > max_reward:
                max_reward = reward
            
            if terminated:
                break
        
        return max_reward + sum_reward, dots_transition_only_result, dots_transition_3D_list, title_for_dots_transition_3D_list
        # return max_reward, dots_transition
        # return sum_reward, dots_transition