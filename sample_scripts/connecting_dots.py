import numpy as np
import matplotlib.pyplot as plt
from functions import UI_dots as ui
from functions import engine_dots as eg

dots_kind_matrix = eg.generate_random_dots(num_dummy_kind=0,num_kind=3, num_layer=2)
dots_kind_matrix = eg.fall_dots_once(dots_kind_matrix)

if_disp = False
if dots_kind_matrix.shape[2]<3:
    if_disp = True
    
if if_disp:
    ui.animate_dots_no_motion(dots_kind_matrix)
    plt.show(block=False)
    ui.print_dots(dots_kind_matrix)

import time

print("##############################")
start = time.time()
connected_dots_list_3D, max_connected_num = eg.connect_dots(dots_kind_matrix, if_only_toBeDeleted=False)
print("time: ", end="")
print(time.time() - start)
if if_disp:
    for ii in range(len(connected_dots_list_3D)):
        print("at layer: {}, max_connected_num: {}".format(ii, max_connected_num[ii]))
        connected_dots_list = connected_dots_list_3D[ii]
        for connected_dots in connected_dots_list:
            if connected_dots.shape[1]>4-1:
                print(connected_dots)

print("##############################")
start = time.time()
connected_dots_list_3D, max_connected_num = eg.connect_dots(dots_kind_matrix, if_only_toBeDeleted=True)
print("time: ", end="")
print(time.time() - start)
if if_disp:
    for ii in range(len(connected_dots_list_3D)):
        print("at layer: {}, max_connected_num: {}".format(ii, max_connected_num[ii]))
        connected_dots_list = connected_dots_list_3D[ii]
        for connected_dots in connected_dots_list:
            print(connected_dots)
