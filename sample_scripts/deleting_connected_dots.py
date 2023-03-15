import numpy as np
import matplotlib.pyplot as plt
from functions import UI_dots as ui
from functions import engine_dots as eg

dots_kind_matrix = eg.generate_random_dots(num_dummy_kind=1,num_kind=2)
dots_kind_matrix = eg.fall_dots_once(dots_kind_matrix)

# =============================================================================
# # Good dots matrix for debug
# dots_kind_matrix = np.flipud(np.array(\
# [[-1 ,-1 ,-1 ,-1 ,-1],\
#  [-1 ,-1 ,-1 ,-1 ,2],\
#  [ 0 ,-1 ,0 ,-1 ,2],\
#  [ 1 ,-1 ,1 ,-1 ,2],\
#  [ 2 ,2 ,2 ,-1 ,0],\
#  [ 1 ,1 ,0 ,0 ,1],\
#  [ 1 ,1 ,1 ,0 ,2],\
#  [ 2 ,2 ,1 ,2 ,0],\
#  [ 0 ,2 ,0 ,0 ,1],\
#  [ 0 ,2 ,0 ,1 ,0]]))
# =============================================================================

ui.print_dots(dots_kind_matrix)
ui.animate_dots_no_motion(dots_kind_matrix)
plt.show(block=False)

connected_dots_list, max_connected_num = eg.connect_dots(dots_kind_matrix)

# Sample for deleting
dots_kind_matrix_deleted = np.copy(dots_kind_matrix)
dots_kind_matrix_deleted = eg.delete_connected_dots(dots_kind_matrix_deleted, connected_dots_list)

ui.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_deleted])
plt.show(block=False)

# Sample for deleting and falling transition
dots_kind_matrix_returned = np.copy(dots_kind_matrix)
dots_kind_matrix_returned, _, _ = \
    eg.delete_and_fall_dots(dots_kind_matrix_returned, connected_dots_list, 4)
    
ui.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_deleted, dots_kind_matrix_returned])
plt.show(block=False)

# Sample for loop of deleting and falling transition
dots_transition, loop_num = eg.delete_and_fall_dots_to_the_end(dots_kind_matrix, 4)

ui.animate_dots_no_motion(dots_transition)
plt.show(block=False)
print("loop_num: "+str(loop_num))