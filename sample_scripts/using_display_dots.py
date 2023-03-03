import os

print('getcwd:      ', os.getcwd())
print('__file__:    ', __file__)

# import modules
import numpy as np
import matplotlib.pyplot as scat

num_horizontal = 5
num_vertical = 10
num_kind = 4
num_dummy_kind = 1

from functions import interface_dots

dots_kind = np.random.randint(0, num_kind + num_dummy_kind + 1,(num_vertical, num_horizontal))
dots_kind = dots_kind.astype(np.float32)
dots_kind[dots_kind>num_kind]=np.nan

interface_dots.print_dots(dots_kind)
interface_dots.scat_dots(dots_kind)

import matplotlib.pyplot as plt
plt.show()