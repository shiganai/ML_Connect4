import numpy as np
import matplotlib.pyplot as plt
from functions import UI_dots as ui
from functions import engine_dots as eg

if_disp_all = False

dots_kind_matrix = eg.generate_random_dots(num_dummy_kind=1,num_kind=3, num_layer=1)
dots_kind_matrix = eg.fall_dots_once(dots_kind_matrix)

ui.print_dots(dots_kind_matrix)

dots_kind_matrix_3D = eg.get_UD_candidate_3D(dots_kind_matrix)

ui.animate_dots_no_motion(np.concatenate([dots_kind_matrix, dots_kind_matrix_3D], axis=2))