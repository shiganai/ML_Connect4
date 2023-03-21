import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import engine_dots as eg

colors = eg.colors
# colors_exist = eg.colors_exist

size = 172 * 0.8 # Set scatter size. 

def print_dots(dots_kind_matrix_3D):
    dots_kind_matrix_3D = eg.convet_2D_dots_to_3D(dots_kind_matrix_3D)
    dots_kind_matrix_3D = np.flip(dots_kind_matrix_3D, axis=0)
    print(dots_kind_matrix_3D.transpose(2,0,1))

def generate_default_figure():
    return plt.figure(figsize=(6.4,4.8))



def scat_dots_by_procedure(dots_kind_matrix_3D_list, title_for_dots_transition_3D_list=None):
    # dots_kind_matrix_3D_list contains several ? x ? x ? ndarray
    
    num_vertical, num_horizontal, num_layer = eg.get_base_dots_info(dots_kind_matrix_3D_list[0])
    x_mesh, y_mesh = np.meshgrid(range(num_horizontal), range(num_vertical))
    
    scat_dots = lambda ax, dots_kind_matrix, title_str: \
                    scat_dots_basic(ax, dots_kind_matrix,\
                    num_horizontal, num_vertical, \
                    x_mesh, y_mesh, size,\
                    title_str)
    
    subplot_row_num = len(dots_kind_matrix_3D_list)
    subplot_col_num = 0
    for list_index in range(subplot_row_num):
        if subplot_col_num < dots_kind_matrix_3D_list[list_index].shape[2]:
            subplot_col_num = dots_kind_matrix_3D_list[list_index].shape[2]
    
    fig = plt.figure(figsize=(6.4*subplot_col_num/2,4.8*subplot_row_num))
    # titleと紐づけ用のカウントを用意.
    ax_count = -1
    for row_num in range(subplot_row_num):
        dots_kind_matrix_3D = dots_kind_matrix_3D_list[row_num]
        for col_num in range(dots_kind_matrix_3D.shape[2]):
            # 一回プロットする前に紐づけカウントを増やしておく.
            ax_count += 1
            frame_index = row_num*subplot_col_num + col_num
            current_ax = fig.add_subplot(subplot_row_num, subplot_col_num, frame_index+1)
            if not(title_for_dots_transition_3D_list is None):
                # title_for_dots_transition_3D_list が入力されていれば ax_count にしたがって利用.
                title_str = title_for_dots_transition_3D_list[ax_count]
            else:
                title_str = None
            scat_dots(current_ax, dots_kind_matrix_3D[:,:,col_num], title_str)


def animate_dots_no_motion(dots_kind_matrix_3D, fig=None, ax=None, mode='subplot'):
    
    dots_kind_matrix_3D = eg.convet_2D_dots_to_3D(dots_kind_matrix_3D)
        
    num_vertical, num_horizontal, num_layer = eg.get_base_dots_info(dots_kind_matrix_3D)
    x_mesh, y_mesh = np.meshgrid(range(num_horizontal), range(num_vertical))
    
    scat_dots = lambda ax, dots_kind_matrix, title_str=None: \
                    scat_dots_basic(ax, dots_kind_matrix,\
                    num_horizontal, num_vertical, \
                    x_mesh, y_mesh, size,\
                    title_str=None)
    
    if mode =='subplot':
        fig, ax, container, = scat_dots_multi_subplot( \
                            fig=fig, \
                            ax=ax, \
                            dots_kind_matrix_3D=dots_kind_matrix_3D, \
                            scat_dots=scat_dots )
        return fig, ax, container,
    else:
        if fig is None:
            fig = generate_default_figure()
        ax = fig.add_subplot(1,1,1)
        if mode == 'anime:func':
            anime = anime_funcUpdate(fig, ax, dots_kind_matrix_3D, scat_dots)
        return fig, ax, anime, 
    
    
    
def scat_dots_multi_subplot(fig, ax, dots_kind_matrix_3D, scat_dots):
    subplot_num = dots_kind_matrix_3D.shape[2]
    subplot_col_num = 6
    subplot_row_num = int( np.ceil(subplot_num / subplot_col_num) )
    
    if fig is None:
        if subplot_num > 2:
            fig = plt.figure(figsize=(6.4*subplot_col_num/2,4.8*subplot_row_num))
        else:
            fig = generate_default_figure()
            subplot_col_num = subplot_num
    else:
        subplot_col_num = subplot_num
    
    is_ax_given = not(ax is None)
    if not(is_ax_given):
        ax = []
    else:
        current_ax = ax
        
    container = []
    for frame_index in range( subplot_num ):
        if not(is_ax_given):
            current_ax = fig.add_subplot(subplot_row_num, subplot_col_num, frame_index+1)
            ax.append( current_ax )
        
        if subplot_num == 1:
            title_str = None
        else:
            title_str = str(frame_index)
            
        container.append(scat_dots(current_ax, dots_kind_matrix_3D[:,:,frame_index], title_str))
        
    return fig, ax, container, 
    
def anime_funcUpdate(fig, ax, dots_kind_matrix_3D, scat_dots):
    if fig is None:
        fig = generate_default_figure()

    def update_frame(frame_index):
        ax.cla()
        title_str = "frame_index = "+str(frame_index)
        scat_dots(ax, dots_kind_matrix_3D[:,:,frame_index], title_str)
    
    anime = animation.FuncAnimation(fig=fig, func=update_frame, frames=2,interval=1000)
    return anime







def scat_dots_basic(ax, dots_kind_matrix,\
                    num_horizontal, num_vertical, \
                    x_mesh, y_mesh, size,\
                    title_str=None):
    ax.axis([-1, num_horizontal, -1, num_vertical]) # Set axis limit
    ax.set_aspect(1) # Normalize the length in the figure
    if not(title_str is None):
        ax.set_title(title_str, fontsize=20)
    
    up_connected_matrix = eg.connect_dots_up(dots_kind_matrix)
    right_connected_matrix = eg.connect_dots_right(dots_kind_matrix)
    
    # 基本的に3Dで帰ってくるから2Dに変換しなおし
    up_connected_matrix = up_connected_matrix[:,:,0]
    right_connected_matrix = right_connected_matrix[:,:,0]

    # The loop below is required because plt.plot does not support multi color input.
    # When you don't care about bond color, you can use the blocked code underneath
    container = []
    for plotting_color_index in range(1, dots_kind_matrix.max()+1):
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
                ax.plot(connection_list_horizontal, connection_list_vertical, c=colors[plotting_color_index], lw=4)\
                )
    
    container.append(\
        ax.scatter(x_mesh, y_mesh, s=size, c=np.array(colors)[dots_kind_matrix.flatten()])\
        )
        
    return container
