def generate_random_dots(num_horizontal=5, num_vertical=10, num_kind=4,num_dummy_kind=1):
    import numpy as np
    dots_kind_matrix = np.random.randint(0, num_kind + num_dummy_kind + 1,(num_vertical, num_horizontal))
    dots_kind_matrix = dots_kind_matrix.astype(np.float32)
    dots_kind_matrix[dots_kind_matrix>num_kind]=np.nan
    return dots_kind_matrix

def print_dots(dots_kind_matrix):
    print(dots_kind_matrix)

def scat_dots(dots_kind_matrix):
    import numpy as np

    dots_kind_matrix = np.flipud(dots_kind_matrix) # flip ud so that the scatter matches print
    dots_kind_matrix[np.isnan(dots_kind_matrix)] = -1 # replace nan to take max

    max_kinds = dots_kind_matrix.max() # get max for for-loop index
    max_kinds = int(max_kinds) # make the max as int for using range()
    
    # get the shape of box
    num_horizontal = dots_kind_matrix.shape[1]
    num_vertical = dots_kind_matrix.shape[0]

    size = 172 * 1 # set scatter size. 
    #TODO: search scatter normalization, or use circle plot using radius

    import matplotlib.pyplot as scat
    f, ax = scat.subplots()
    for target_kind in range(max_kinds + 1):
        target_kind_yx = np.where(dots_kind_matrix==target_kind) # find indecies of target kinds
        ax.scatter(target_kind_yx[1], target_kind_yx[0], s=size) # scatter. Be aware that the indecies are in shape of (y,x)

    ax.axis([-1, num_horizontal, -1, num_vertical]) # set axis limit
    ax.set_aspect(1) # normalize the length in the figure
