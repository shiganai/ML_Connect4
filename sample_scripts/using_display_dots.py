from functions import interface_dots
dots_kind_matrix = interface_dots.generate_random_dots()
interface_dots.print_dots(dots_kind_matrix)
interface_dots.scat_dots(dots_kind_matrix)

import matplotlib.pyplot as plt
plt.show()