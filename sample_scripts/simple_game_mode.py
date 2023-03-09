import numpy as np
import matplotlib.pyplot as plt
from functions import UI_dots as ui
from functions import engine_dots as eg

# Prepare empty box
num_horizontal = 5
num_vertical = 7
num_kind = 3
dots_kind_matrix = np.full((num_vertical, num_horizontal), -1)

# =============================================================================
# # Prepare rondomly generated dots
# dots_kind_matrix = eg.generate_random_dots(num_horizontal, num_vertical, num_kind=num_kind)
# dots_kind_matrix = eg.fall_dots_once(dots_kind_matrix)
# dots_transition, loop_num = eg.delete_and_fall_dots_to_the_end(dots_kind_matrix, 4)
# dots_kind_matrix = dots_transition[-1]
# =============================================================================

# selected_candidate = input("select candidate number: ")
# dots_kind_matrix = candidate_list[int(selected_candidate)]

while True:
    if_user_input_end = False
    
    next_2dots = np.random.randint(0,num_kind,(2,1))
    
    candidate_list = eg.get_candidate(dots_kind_matrix, next_2dots)
    ui.animate_dots_no_motion(candidate_list)
    plt.show(block=False)

    while True:
        try:
            selected_candidate = input("select candidate number or q to quit: ")
            if selected_candidate=="q":
                if_user_input_end = True
                break
            dots_kind_matrix = candidate_list[int(selected_candidate)]
            break
        except Exception as e:
            print(e)
        
    if if_user_input_end:
        break
    
    dots_transition, loop_num = eg.delete_and_fall_dots_to_the_end(dots_kind_matrix, 4)
    print("loop_num = "+str(loop_num))
    ui.animate_dots_no_motion(dots_transition)
    plt.show(block=False)
    
    dots_kind_matrix = dots_transition[-1]