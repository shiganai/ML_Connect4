from functions import interface_dots
dots_kind_matrix = interface_dots.generate_random_dots(num_dummy_kind=2,num_kind=2)
dots_kind_matrix = interface_dots.fall_dots_once(dots_kind_matrix)

import numpy as np
# dots_kind_matrix = np.flipud(np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,0,-1,-1,-1],[-1,0,-1,-1,0],[0,0,-1,-1,1],[0,0,-1,0,0],[0,1,-1,0,0],[1,0,-1,1,1],[1,0,-1,1,1],[1,0,1,0,1]]))
interface_dots.animate_dots_no_motion(dots_kind_matrix)

import matplotlib.pyplot as plt
plt.show(block=False)

interface_dots.print_dots(dots_kind_matrix)

connected_dots_list, _ = interface_dots.connect_dots(dots_kind_matrix)
for connected_dots in connected_dots_list:
    print(connected_dots)

# interface_dots.animate_dots_no_motion(connected_dots_matrix)
# plt.show(block=False)

# interface_dots.animate_dots_no_motion([dots_kind_matrix, connected_dots_matrix])
# plt.show(block=False)

