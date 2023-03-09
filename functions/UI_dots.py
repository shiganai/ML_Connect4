import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import engine_dots as eg
    
# Initiate colors
colors_exist = ['blue', 'red', 'green', 'purple', 'yellow']
repeat_num = 2
colors = colors_exist*repeat_num
colors.append('none')

connected_threshold_default = 4

def print_dots(dots_kind_matrix):
    print(np.flipud(dots_kind_matrix))

def animate_dots_no_motion(dots_kind_matrix_3D, mode='subplot'):
    
    if type(dots_kind_matrix_3D) is not list:
        dots_kind_matrix_3D = [dots_kind_matrix_3D]
    
    dots_kind_matrix = dots_kind_matrix_3D[0]
    
    size = 172 * 1 # Set scatter size. 

    fig = plt.figure(figsize=(6.4,4.8))
    num_vertical, num_horizontal = eg.get_base_dots_info(dots_kind_matrix)
    x_mesh, y_mesh = np.meshgrid(range(num_horizontal), range(num_vertical))

    def scat_dots(ax, dots_kind_matrix):
        ax.axis([-1, num_horizontal, -1, num_vertical]) # Set axis limit
        ax.set_aspect(1) # Normalize the length in the figure
        up_connected_matrix = eg.connect_dots_up(dots_kind_matrix)
        right_connected_matrix = eg.connect_dots_right(dots_kind_matrix)

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
    subplot_num = len(dots_kind_matrix_3D)
    subplot_col_num = 4
    subplot_row_num = int( np.ceil(subplot_num / subplot_col_num) )
    
    if subplot_num > 2:
        plt.close(fig)
        fig = plt.figure(figsize=(6.4*subplot_col_num/2,4.8*subplot_row_num))
    else:
        subplot_col_num = subplot_num
    
    ax = []
    container = []
    for frame_index in range( subplot_num ):
        ax.append( fig.add_subplot(subplot_row_num, subplot_col_num, frame_index+1) )
        container.append(scat_dots(ax[-1], dots_kind_matrix_3D[frame_index]))
        
    return ax, container, 
    
def anime_funcUpdate(fig, ax, dots_kind_matrix_3D, scat_dots):

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

    artists=[]
    for frame_index in range( len(dots_kind_matrix_3D) ):
        container = scat_dots(ax, dots_kind_matrix_3D[frame_index])
        title = ax.text(ax.get_xlim()[0],ax.get_ylim()[1]*1.05,"frame_index = "+str(frame_index))
        container.append(title)
        artists.append(container)

    anime = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000) # anime is needed to keep animation visually.
    return anime

    