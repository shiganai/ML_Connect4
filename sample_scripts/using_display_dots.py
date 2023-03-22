import matplotlib.pyplot as plt
from functions import UI_dots as ui
from functions import engine_dots as eg

dots_kind_matrix = eg.generate_random_dots()
ui.print_dots(dots_kind_matrix)
anime = ui.animate_dots_no_motion(dots_kind_matrix)

plt.show(block=False)