def print_dots(dots_kind):
    print(dots_kind)

def scat_dots(dots_kind):
    import numpy as np

    dots_kind = np.flipud(dots_kind) # flip ud so that the scatter matches print
    dots_kind[np.isnan(dots_kind)] = -1 # replace nan to take max

    max_kinds = dots_kind.max() # get max for for-loop index
    max_kinds = int(max_kinds) # make the max as int for using range()
    
    # get the shape of box
    num_horizontal = dots_kind.shape[0]
    num_vertical = dots_kind.shape[1]

    size = 172 * 1 # set scatter size. 
    #TODO: search scatter normalization, or use circle plot using radius

    import matplotlib.pyplot as scat
    f, ax = scat.subplots()
    for target_kind in range(max_kinds + 1):
        target_kind_yx = np.where(dots_kind==target_kind) # find indecies of target kinds
        ax.scatter(target_kind_yx[1], target_kind_yx[0], s=size) # scatter. Be aware that the indecies are in shape of (y,x)

    ax.axis([-1, num_vertical, -1, num_horizontal]) # set axis limit
    ax.set_box_aspect(1) # normalize the length in the figure