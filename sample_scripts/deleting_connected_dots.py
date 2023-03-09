import numpy as np
import matplotlib.pyplot as plt
from functions import interface_dots

dots_kind_matrix = interface_dots.generate_random_dots(num_dummy_kind=1,num_kind=2)
dots_kind_matrix = interface_dots.fall_dots_once(dots_kind_matrix)

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

interface_dots.print_dots(dots_kind_matrix)
interface_dots.animate_dots_no_motion(dots_kind_matrix)
plt.show(block=False)

connected_dots_list, max_connected_num = interface_dots.connect_dots(dots_kind_matrix)

# Sample for deleting
dots_kind_matrix_deleted = np.copy(dots_kind_matrix)
dots_kind_matrix_deleted = interface_dots.delete_connected_dots(dots_kind_matrix_deleted, connected_dots_list)

interface_dots.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_deleted])
plt.show(block=False)

# Sample for deleting and falling transition
dots_kind_matrix_returned = np.copy(dots_kind_matrix)
dots_kind_matrix_returned, _, _ = \
    interface_dots.delete_and_fall_dots(dots_kind_matrix_returned, connected_dots_list, 4)
    
interface_dots.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_deleted, dots_kind_matrix_returned])
plt.show(block=False)

# Sample for loop of deleting and falling transition
dots_transition, loop_num = interface_dots.delete_and_fall_dots_to_the_end(dots_kind_matrix, 4)

interface_dots.animate_dots_no_motion(dots_transition)
plt.show(block=False)
print("loop_num: "+str(loop_num))