from functions import interface_dots
dots_kind_matrix = interface_dots.generate_random_dots(num_dummy_kind=3)
interface_dots.print_dots(dots_kind_matrix)

dots_kind_matrix_falled = interface_dots.fall_dots_once(dots_kind_matrix)
interface_dots.print_dots(dots_kind_matrix_falled)

interface_dots.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_falled])

interface_dots.animate_dots_no_motion([dots_kind_matrix, dots_kind_matrix_falled], anime_mode='artists')