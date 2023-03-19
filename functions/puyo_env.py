
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
        self.candidates = eg.get_candidate(\
                              dots_kind_matrix=self.dots_kind_matrix, \
                              next_2dots=self.next_2dots[:,0], \
                              is_ignore_same=True)
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
            for evaled_candidate in self.candidates_dots_result:
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
            
            best_index = best_NN_index
            
            # 想定される連鎖数が最も大きいものを特定
            best_loop_num_value = np.array(self.candidates_loop_num).max()
            best_loop_num_index = np.where(self.candidates_loop_num == best_loop_num_value)[0]
            
            if len(best_loop_num_index) > 1:
                best_loop_num_index_best_NN_index = NN_values[best_loop_num_index].argmax()
                best_loop_num_index = best_loop_num_index[best_loop_num_index_best_NN_index]
                # best_loop_num_index = np.random.randint(0, len(best_loop_num_index))
                
            else:
                best_loop_num_index = best_loop_num_index[0]
            
            if if_disp:
                print("loop_num: {}, NN_value: {:>5.2f}".format(best_loop_num_value, best_NN_value), \
                      end="")
                    
            if best_loop_num_value > 0:
                if best_loop_num_index == best_NN_index:
                    None
                    if if_disp:
                        print(", and they chose the same candidate", end="")
                else:
                    if best_loop_num_value > 9:
                        best_index = best_loop_num_index
                        if if_disp:
                            print(", so chose loop_num", end="")
                    else:
                        if best_loop_num_value > best_NN_value:
                            best_index = best_loop_num_index
                            if if_disp:
                                print(", so chose loop_num", end="")
                        else:
                            if if_disp:
                                print(", so chose NN_value", end="")
            
            if if_disp:
                print(" at turn: " + str(self.turn_count))
                
            if if_disp:
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