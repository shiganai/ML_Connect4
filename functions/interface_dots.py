# initiate colors
colors = ['blue', 'red', 'green', 'purple', 'yellow', 'none']

def generate_random_dots(num_horizontal=5, num_vertical=10, num_kind=3,num_dummy_kind=2):
    import warnings
    if num_kind > colors.__len__():
        warnings.warn('num_kind is supported up to 5. The color will be duplicated')
    import numpy as np
    dots_kind_matrix = np.random.randint(0, num_kind + num_dummy_kind + 1,(num_vertical, num_horizontal))
    dots_kind_matrix[dots_kind_matrix>num_kind] = -1
    return dots_kind_matrix

def print_dots(dots_kind_matrix):
    print(dots_kind_matrix)

def animate_dots_no_motion(dots_kind_matrix_3D, anime_mode='func'):
    is_animating = True
    if type(dots_kind_matrix_3D) is not list:
        dots_kind_matrix_3D = [dots_kind_matrix_3D]
        is_animating = False
    
    import numpy as np

    dots_kind_matrix = dots_kind_matrix_3D[0]

    max_kinds = dots_kind_matrix.max() # get max for for-loop index
    max_kinds = int(max_kinds) # make the max as int for using range()
    
    # get the shape of box
    num_horizontal = dots_kind_matrix.shape[1]
    num_vertical = dots_kind_matrix.shape[0]

    size = 172 * 1 # set scatter size. 
    #TODO: search scatter normalization, or use circle plot using radius

    import matplotlib.pyplot as plt
    # import matplotlib.animation as animation

    fig, ax = plt.subplots()
    ax.axis([-1, num_horizontal, -1, num_vertical]) # set axis limit
    ax.set_aspect(1) # normalize the length in the figure
    x_mesh, y_mesh = np.meshgrid(range(num_horizontal), range(num_vertical))

    def scat_dots(dots_kind_matrix):
        dots_kind_matrix = np.flipud(dots_kind_matrix)
        return ax.scatter(x_mesh, y_mesh, s=size, c=np.array(colors)[dots_kind_matrix.flatten()])
    
    if is_animating:
        if anime_mode == 'func':
            anime = anime_funcUpdate(fig, ax, dots_kind_matrix_3D, scat_dots)
        elif anime_mode == 'artists':
            anime = anime_artists(fig, ax, dots_kind_matrix_3D, scat_dots)
        return anime
    else:
        container = scat_dots(dots_kind_matrix)
        return container
    
def anime_funcUpdate(fig, ax, dots_kind_matrix_3D, scat_dots):
    import matplotlib.animation as animation

    def update_frame(frame_index):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.cla()
        ax.set_title("frame_index = "+str(frame_index))
        scat_dots(dots_kind_matrix_3D[frame_index])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    anime = animation.FuncAnimation(fig=fig, func=update_frame, frames=2,interval=1000)
    return anime
    # plt.show()

def anime_artists(fig, ax, dots_kind_matrix_3D, scat_dots):
    import matplotlib.animation as animation

    artists=[]
    for frame_index in range( dots_kind_matrix_3D.__len__() ):
        title = ax.text(ax.get_xlim()[0],ax.get_ylim()[1]*1.05,"frame_index = "+str(frame_index))
        container = scat_dots(dots_kind_matrix_3D[frame_index])
        artists.append([container,title])

    anime = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000) # anime is needed to keep animation visually.
    return anime
    # plt.show()

def fall_dots_once(dots_kind_matrix):
    import numpy as np

    dots_kind_matrix_falled = np.copy(dots_kind_matrix)

    is_empty_matrix = dots_kind_matrix > -1
    empty_sorted_indecies = np.argsort(is_empty_matrix,axis=0)

    # get the shape of box
    num_horizontal = dots_kind_matrix.shape[1]

    for target_horizontal_index in range(num_horizontal):
        target_vertical_vector = dots_kind_matrix_falled[:,target_horizontal_index]
        target_vertical_vector = target_vertical_vector[empty_sorted_indecies[:,target_horizontal_index]]
        # print(target_vertical_vector)
        dots_kind_matrix_falled[:,target_horizontal_index] = target_vertical_vector
    return dots_kind_matrix_falled

def connect_dots(dots_kind_matrix):
    import numpy as np

    empty_matrix = dots_kind_matrix == -1
    print(empty_matrix)