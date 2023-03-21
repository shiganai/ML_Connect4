import numpy as np
import matplotlib.pyplot as plt
from functions import UI_dots as ui
from functions import engine_dots as eg

if_disp_all = False

dots_kind_matrix = eg.generate_random_dots(num_dummy_kind=0,num_kind=3, num_layer=5)
dots_kind_matrix = eg.fall_dots_once(dots_kind_matrix)

if if_disp_all:
    ui.print_dots(dots_kind_matrix)
    ui.animate_dots_no_motion(dots_kind_matrix)
    plt.show(block=False)

connected_dots_list_3D, max_connected_num = eg.connect_dots(dots_kind_matrix)

# Sample for deleting
dots_kind_matrix_deleted = np.copy(dots_kind_matrix)
dots_kind_matrix_deleted = eg.delete_connected_dots(dots_kind_matrix_deleted, connected_dots_list_3D)

if if_disp_all:
    for layer_index in range(dots_kind_matrix.shape[2]):
        animating_dots_kind_mat = np.concatenate(\
                                           [dots_kind_matrix[:,:,layer_index,np.newaxis], \
                                            dots_kind_matrix_deleted[:,:,layer_index, np.newaxis]
                                            ], \
                                           axis=2)
        _,_,container = ui.animate_dots_no_motion(animating_dots_kind_mat)
        plt.show(block=False)

# Sample for deleting and falling transition
dots_kind_matrix_result = np.copy(dots_kind_matrix)
dots_kind_matrix_result, _, _ = \
    eg.delete_and_fall_dots(dots_kind_matrix_result, connected_dots_list_3D, 4)

if if_disp_all:
    for layer_index in range(dots_kind_matrix_result.shape[2]):
        animating_dots_kind_mat = np.concatenate([\
                                                  dots_kind_matrix[:,:,layer_index,np.newaxis], \
                                                  dots_kind_matrix_deleted[:,:,layer_index, np.newaxis], \
                                                  dots_kind_matrix_result[:,:,layer_index, np.newaxis], \
                                                  ], \
                                                 axis=2)
        _,_,container = ui.animate_dots_no_motion(animating_dots_kind_mat)
        plt.show(block=False)

# Sample for loop of deleting and falling transition
dots_kind_matrix_result, loop_num, dots_transition = \
    eg.delete_and_fall_dots_to_the_end(dots_kind_matrix, if_return_only_result=False)

if if_disp_all:
    for layer_index in range(dots_kind_matrix_result.shape[2]):
        animating_dots_kind_mat = dots_transition[layer_index]
        _,_,container = ui.animate_dots_no_motion(animating_dots_kind_mat)
        plt.show(block=False)
        

ui.scat_dots_by_procedure(dots_transition)