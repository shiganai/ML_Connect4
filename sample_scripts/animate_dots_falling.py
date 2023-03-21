import numpy as np
import matplotlib.pyplot as plt
from functions import UI_dots as ui
from functions import engine_dots as eg

dots_kind_matrix = eg.generate_random_dots(num_dummy_kind=1, num_layer=2)
ui.print_dots(dots_kind_matrix)

dots_kind_matrix_falled = np.copy(dots_kind_matrix)
dots_kind_matrix_falled = eg.fall_dots_once(dots_kind_matrix_falled)
ui.print_dots(dots_kind_matrix_falled)

for layer_index in range(dots_kind_matrix_falled.shape[2]):
    animating_dots_kind_mat = np.concatenate(\
                                       [dots_kind_matrix[:,:,layer_index,np.newaxis], \
                                        dots_kind_matrix_falled[:,:,layer_index, np.newaxis]
                                        ], \
                                       axis=2)
    _,_,container = ui.animate_dots_no_motion(animating_dots_kind_mat)
    plt.show(block=False)

    _,_,anime = ui.animate_dots_no_motion(animating_dots_kind_mat, mode='anime:func')
    plt.show(block=False)
    plt.pause(0.1)