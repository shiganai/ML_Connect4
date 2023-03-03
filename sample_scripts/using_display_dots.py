from functions import interface_dots
dots_kind_matrix = interface_dots.generate_random_dots()
interface_dots.print_dots(dots_kind_matrix)
interface_dots.animate_dots_no_motion(dots_kind_matrix)

import matplotlib.pyplot as plt
plt.show()