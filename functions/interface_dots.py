# Initiate colors
import numpy as np
colors_exist = ['blue', 'red', 'green', 'purple', 'yellow']
repeat_num = 2
# colors = np.hstack([np.repeat(colors_exist,repeat_num), 'none'])
colors = colors_exist*repeat_num
colors.append('none')

def generate_random_dots(num_horizontal=5, num_vertical=10, num_kind=3,num_dummy_kind=2):
    import warnings
    if num_kind > colors.__len__():
        warnings.warn('num_kind is supported up to 5. The color will be duplicated')
    import numpy as np
    dots_kind_matrix = np.random.randint(0, num_kind + num_dummy_kind + 1,(num_vertical, num_horizontal))
    dots_kind_matrix[dots_kind_matrix>num_kind] = -1
    return dots_kind_matrix

def print_dots(dots_kind_matrix):
    import numpy as np
    print(np.flipud(dots_kind_matrix))

def get_base_dots_info(dots_kind_matrix):
    # Get the shape of box
    num_vertical = dots_kind_matrix.shape[0]
    num_horizontal = dots_kind_matrix.shape[1]
    return num_vertical, num_horizontal

def animate_dots_no_motion(dots_kind_matrix_3D, mode='subplot'):
    if type(dots_kind_matrix_3D) is not list:
        dots_kind_matrix_3D = [dots_kind_matrix_3D]
    

    dots_kind_matrix = dots_kind_matrix_3D[0]
    
    size = 172 * 1 # Set scatter size. 

    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure()
    num_vertical, num_horizontal = get_base_dots_info(dots_kind_matrix)
    x_mesh, y_mesh = np.meshgrid(range(num_horizontal), range(num_vertical))

    def scat_dots(ax, dots_kind_matrix):
        ax.axis([-1, num_horizontal, -1, num_vertical]) # Set axis limit
        ax.set_aspect(1) # Normalize the length in the figure
        up_connected_matrix = connect_dots_up(dots_kind_matrix)
        right_connected_matrix = connect_dots_right(dots_kind_matrix)

        # The loop below is required because plt.plot does not support multi color input.
        # When you don't care about bond color, you can use the blocked code underneath
        container = []
        for plotting_color_index in range(dots_kind_matrix.max()+1):
            connection_vertical = np.where(up_connected_matrix & (dots_kind_matrix == plotting_color_index))
            connection_horizontal = np.where(right_connected_matrix & (dots_kind_matrix == plotting_color_index))
            
            connection_list_horizontal = \
                [np.hstack( [ connection_vertical[1], connection_horizontal[1]   ]), \
                  np.hstack( [ connection_vertical[1], connection_horizontal[1]+1 ])]
                    
            connection_list_vertical = \
                [np.hstack( [ connection_vertical[0]      , connection_horizontal[0] ] ), \
                  np.hstack( [ connection_vertical[0] + 1  , connection_horizontal[0] ] )]
            
            if connection_list_horizontal[0].size > 0:
                container.append(\
                    ax.plot(connection_list_horizontal, connection_list_vertical, c=colors[plotting_color_index])\
                    )
            
# =============================================================================
#         connection_vertical = np.where(up_connected_matrix)
#         connection_horizontal = np.where(right_connected_matrix)
#         
#         connection_list_horizontal = \
#             [np.hstack( [ connection_vertical[1], connection_horizontal[1]   ]), \
#              np.hstack( [ connection_vertical[1], connection_horizontal[1]+1 ])]
#                 
#         connection_list_vertical = \
#             [np.hstack( [ connection_vertical[0]      , connection_horizontal[0] ] ), \
#              np.hstack( [ connection_vertical[0] + 1  , connection_horizontal[0] ] )]
#         
#         if connection_list_horizontal[0].size > 0:
#             container = ax.plot(connection_list_horizontal, connection_list_vertical, c='black' )
# =============================================================================
        
        container.append(\
            ax.scatter(x_mesh, y_mesh, s=size, c=np.array(colors)[dots_kind_matrix.flatten()])\
            )
            
        return container
    
    if mode =='subplot':
        ax, container, = scat_dots_multi_subplot(fig, dots_kind_matrix_3D, scat_dots)
        return fig, ax, container,
    else:
        ax = fig.add_subplot(1,1,1)
        if mode == 'anime:func':
            anime = anime_funcUpdate(fig, ax, dots_kind_matrix_3D, scat_dots)
        elif mode == 'anime:artists':
            anime = anime_artists(fig, ax, dots_kind_matrix_3D, scat_dots)
        return fig, ax, anime, 

def scat_dots_multi_subplot(fig, dots_kind_matrix_3D, scat_dots):
    ax = []
    container = []
    for frame_index in range( dots_kind_matrix_3D.__len__() ):
        ax.append( fig.add_subplot(1, dots_kind_matrix_3D.__len__(), frame_index+1) )
        container.append(scat_dots(ax[-1], dots_kind_matrix_3D[frame_index]))
        
    return ax, container, 
    
def anime_funcUpdate(fig, ax, dots_kind_matrix_3D, scat_dots):
    import matplotlib.animation as animation

    def update_frame(frame_index):
        ax.cla()
        ax.set_title("frame_index = "+str(frame_index))
        scat_dots(ax, dots_kind_matrix_3D[frame_index])
    
    anime = animation.FuncAnimation(fig=fig, func=update_frame, frames=2,interval=1000)
    return anime

def anime_artists(fig, ax, dots_kind_matrix_3D, scat_dots):
    raise Exception('This function is abondoned. See comment below')
# =============================================================================
#     Adding the connection bond visually by plt.plot, 
#     Python started to throw an error saying
#     "artist.set_visible(False)"
#     "AttributeError: 'list' object has no attribute 'set_visible'"
#     at anime = animation.ArtistAnimation ... code.
#     I could not solve this issue.
# =============================================================================
    import matplotlib.animation as animation

    artists=[]
    for frame_index in range( dots_kind_matrix_3D.__len__() ):
        container = scat_dots(ax, dots_kind_matrix_3D[frame_index])
        title = ax.text(ax.get_xlim()[0],ax.get_ylim()[1]*1.05,"frame_index = "+str(frame_index))
        container.append(title)
        artists.append(container)

    anime = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000) # anime is needed to keep animation visually.
    return anime

def fall_dots_once(dots_kind_matrix):
    import numpy as np

    dots_kind_matrix_falled = np.copy(dots_kind_matrix)

    is_empty_matrix = dots_kind_matrix < 0
    empty_sorted_indecies = np.argsort(is_empty_matrix,axis=0)

    # Get the shape of box
    num_horizontal = dots_kind_matrix.shape[1]

    for target_horizontal_index in range(num_horizontal):
        target_vertical_vector = dots_kind_matrix_falled[:,target_horizontal_index]
        target_vertical_vector = target_vertical_vector[empty_sorted_indecies[:,target_horizontal_index]]
        # print(target_vertical_vector)
        dots_kind_matrix_falled[:,target_horizontal_index] = target_vertical_vector
    return dots_kind_matrix_falled

def connect_dots(dots_kind_matrix):
    import numpy as np
    # Use while true loop for checking dots connection

    is_checked_matrix = np.full_like(dots_kind_matrix, False, dtype=bool) # when this value is true, it is already checked and no more checking is needed.
    empty_matrix = dots_kind_matrix == -1 # get the empty cells to replace is_checked
    is_checked_matrix[empty_matrix] = True

    up_connected_matrix = connect_dots_up(dots_kind_matrix)
    right_connected_matrix = connect_dots_right(dots_kind_matrix)
    
    # Get the shape of box
    num_vertical = dots_kind_matrix.shape[0]
    num_horizontal = dots_kind_matrix.shape[1]

    connected_dots_list = [] # Initiate a list to hold connecting info
    

    for target_horizontal_index in range(num_horizontal):
        for target_vertical_index in range(num_vertical):
            # Start while loop until the connection ends up?
            adding_connected_dots = np.array([target_vertical_index, target_horizontal_index, False])

            while True:
                # Search not checked adding_connected_dots from bottom, except for the first loop
                # Refresh checking_dots_hor based on checking_adding_connected_dots_index
                if adding_connected_dots.ndim==1:
                    # Check if the initial dot is already checked. This sometimes happen when the dot is lonly.
                    if adding_connected_dots[2] == 1:
                        break

                    checking_adding_connected_dots_index = 0
                    checking_dots_ver = adding_connected_dots[0]
                    checking_dots_hor = adding_connected_dots[1]
                    adding_connected_dots[2] = True
                else:
                    not_yet_checked_index = np.where(adding_connected_dots[:,2]==0)
                    if len(not_yet_checked_index[0]) == 0:
                        adding_connected_dots = np.unique(adding_connected_dots,axis=0)
                        adding_connected_dots = np.array([adding_connected_dots[:,0], adding_connected_dots[:,1]]) # Remove is_checked_info
                        connected_dots_list.append(adding_connected_dots)
                        break
                    
                    checking_adding_connected_dots_index = not_yet_checked_index[0][-1]
                    checking_dots_ver = adding_connected_dots[checking_adding_connected_dots_index,0]
                    checking_dots_hor = adding_connected_dots[checking_adding_connected_dots_index,1]
                    adding_connected_dots[checking_adding_connected_dots_index,2] = True

                # Check if the target dot has been already checked.
                if ~is_checked_matrix[checking_dots_ver, checking_dots_hor]:
                    # When the target dot is to be checked whether it is conncedted up or right
                    is_checked_matrix[checking_dots_ver, checking_dots_hor] = True

                    is_nonchecked_up_connected = \
                        up_connected_matrix[checking_dots_ver, checking_dots_hor] \
                            and ~is_checked_matrix[checking_dots_ver + 1, checking_dots_hor]
                    is_nonchecked_right_connected = \
                        right_connected_matrix[checking_dots_ver, checking_dots_hor] \
                            and ~is_checked_matrix[checking_dots_ver, checking_dots_hor + 1]
                    is_nonchecked_down_connected = \
                        up_connected_matrix[checking_dots_ver-1, checking_dots_hor] \
                            and ~is_checked_matrix[checking_dots_ver-1, checking_dots_hor]

                    if is_nonchecked_right_connected:
                        # When the target dot is connected to right, add the right dots to the list
                        adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver, checking_dots_hor + 1, False]])

                    if is_nonchecked_up_connected:
                        # When the target dot is connected to upper, add the upeer dots to the list
                        adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver + 1, checking_dots_hor, False]])

                    if is_nonchecked_down_connected:
                        # When the target dot is connected to upper, add the upeer dots to the list
                        adding_connected_dots = np.vstack([adding_connected_dots,[checking_dots_ver - 1, checking_dots_hor, False]])

    return connected_dots_list

def connect_dots_up(dots_kind_matrix):
    import numpy as np

    empty_matrix = dots_kind_matrix == -1 # Get the empty cells to replace later
    diff_up_matrix = np.vstack([dots_kind_matrix[1:,:] - dots_kind_matrix[0:-1,:], np.ones((1,dots_kind_matrix.shape[1]))]) # if this value is 0, the kinds of dot are the same between upper and lower
    # Note that 1 are inserted at the bottom cells
    
    diff_up_matrix[empty_matrix] = 1 # replace empty cells as 1, meanning not connected
    up_connected_matrix = diff_up_matrix == 0 # Get the upper connected cells
    return up_connected_matrix

def connect_dots_right(dots_kind_matrix):
    import numpy as np

    empty_matrix = dots_kind_matrix == -1 # Get the empty cells to replace later
    diff_right_matrix = np.hstack([dots_kind_matrix[:,1:] - dots_kind_matrix[:,0:-1], np.ones((dots_kind_matrix.shape[0],1))]) # if this value is 0, the kinds of dot are the same between upper and lower
    # Note that 1 are inserted at the most right cells
    
    diff_right_matrix[empty_matrix] = 1 # Replace empty cells as 1, meanning not connected
    right_connected_matrix = diff_right_matrix == 0 # Get the upper connected cells
    return right_connected_matrix