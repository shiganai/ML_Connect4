from functions import interface_dots
dots_kind_matrix = interface_dots.generate_random_dots(num_dummy_kind=3)
interface_dots.print_dots(dots_kind_matrix)

dots_kind_matrix_falled = interface_dots.fall_dots_once(dots_kind_matrix)
interface_dots.print_dots(dots_kind_matrix_falled)

interface_dots.animate_dots_no_motion(dots_kind_matrix_falled)
import matplotlib.pyplot as plt
plt.show(block=False)

_,_,container = interface_dots.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_falled])
plt.show(block=False)

_,_,anime = interface_dots.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_falled], mode='anime:func')
plt.show(block=False)
plt.pause(0.1)

_,_,anime = interface_dots.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_falled], mode='anime:artists')
plt.show(block=False)
plt.pause(0.1)