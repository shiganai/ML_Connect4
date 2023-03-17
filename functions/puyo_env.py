
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import UI_dots as ui
from functions import engine_dots as eg

class puyo_env:
    
    # def __init__(self):
                
    def __init__(self, \
            dots_kind_matrix=None, \
            num_horizontal = eg.num_horizontal_default, \
            num_vertical = eg.num_vertical_default, \
            num_kind = eg.num_kind_default, \
            num_dummy_kind = eg.num_dummy_kind_default):
            
        print("construct of puyo_env is called")

        self.num_horizontal = num_horizontal
        self.num_vertical = num_vertical
        self.num_kind = num_kind
        self.num_dummy_kind = num_dummy_kind
        self.num_candidate = self.num_horizontal * 2 + (self.num_horizontal-1) * 2
        
        self.action_space = self.action_space(self)
        
        self.turn_count_threshold = 100
    
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
        
        reward = float(loop_num * loop_num)
        
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
        self.next_2dots = np.random.randint(1,self.num_kind+1,(2,3))
    
    def update_next_2dots(self):
        adding_2dots = np.random.randint(1,self.num_kind+1,(1,2))
        self.next_2dots = np.transpose(np.vstack([self.next_2dots[:,1], self.next_2dots[:,2], adding_2dots]))
        
    def update_state(self):
        dots_kind_matrix = np.copy(self.dots_kind_matrix)
        
        horizontal_index = 0
        dots_kind_matrix[-2,horizontal_index] = self.next_2dots[0,horizontal_index]
        dots_kind_matrix[-1,horizontal_index] = self.next_2dots[1,horizontal_index] 
        
        horizontal_index = 1
        dots_kind_matrix[-2,horizontal_index] = self.next_2dots[0,horizontal_index]
        dots_kind_matrix[-1,horizontal_index] = self.next_2dots[1,horizontal_index] 
        
        horizontal_index = 2
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
            dots_transition, loop_num = eg.delete_and_fall_dots_to_the_end(self.dots_kind_matrix, 4)
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