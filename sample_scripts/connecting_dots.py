from functions import interface_dots
dots_kind_matrix = interface_dots.generate_random_dots(num_dummy_kind=2,num_kind=1)
dots_kind_matrix = interface_dots.fall_dots_once(dots_kind_matrix)

import numpy as np
dots_kind_matrix = np.flipud(np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,0,-1,-1,-1],[-1,0,-1,-1,0],[0,0,-1,-1,1],[0,0,-1,0,0],[0,1,-1,0,0],[1,0,-1,1,1],[1,0,-1,1,1],[1,0,1,0,1]]))
interface_dots.animate_dots_no_motion(dots_kind_matrix)

interface_dots.print_dots(dots_kind_matrix)


import numpy as np
# not_empty_matrix = ~(dots_kind_matrix == -1)
# a = np.vstack([np.ones((1,dots_kind_matrix.shape[1])), dots_kind_matrix[0:-1,:] - dots_kind_matrix[1:,:]])

# up_connected_matrix = interface_dots.connect_dots_up(dots_kind_matrix)
# print(up_connected_matrix)

# right_connected_matrix = interface_dots.connect_dots_right(dots_kind_matrix)
# print(right_connected_matrix)

connected_dots_matrix, connected_dots_list = interface_dots.connect_dots(dots_kind_matrix)

interface_dots.animate_dots_no_motion(connected_dots_matrix)



import matplotlib.pyplot as plt
plt.show()