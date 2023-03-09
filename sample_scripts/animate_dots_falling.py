import numpy as np
import matplotlib.pyplot as plt
from functions import UI_dots as ui
from functions import engine_dots as eg

dots_kind_matrix = eg.generate_random_dots(num_dummy_kind=3)
ui.print_dots(dots_kind_matrix)

dots_kind_matrix_falled = np.copy(dots_kind_matrix)
dots_kind_matrix_falled = eg.fall_dots_once(dots_kind_matrix_falled)
ui.print_dots(dots_kind_matrix_falled)

ui.animate_dots_no_motion(dots_kind_matrix_falled)
plt.show(block=False)

_,_,container = ui.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_falled])
plt.show(block=False)

_,_,anime = ui.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_falled], mode='anime:func')
plt.show(block=False)
plt.pause(0.1)

# mode anime:artists is abondoned.
# =============================================================================
# _,_,anime = ui.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_falled], mode='anime:artists')
# plt.show(block=False)
# plt.pause(0.1)
# =============================================================================
