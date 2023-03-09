import numpy as np
import matplotlib.pyplot as plt
from functions import UI_dots as ui
from functions import engine_dots as eg

dots_kind_matrix = eg.generate_random_dots(num_dummy_kind=2,num_kind=2)
dots_kind_matrix = eg.fall_dots_once(dots_kind_matrix)

# dots_kind_matrix = np.flipud(np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,0,-1,-1,-1],[-1,0,-1,-1,0],[0,0,-1,-1,1],[0,0,-1,0,0],[0,1,-1,0,0],[1,0,-1,1,1],[1,0,-1,1,1],[1,0,1,0,1]]))
ui.animate_dots_no_motion(dots_kind_matrix)

plt.show(block=False)

ui.print_dots(dots_kind_matrix)

connected_dots_list, _ = eg.connect_dots(dots_kind_matrix)
for connected_dots in connected_dots_list:
    print(connected_dots)
