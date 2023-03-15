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

def generate_default_figure():
    return plt.figure(figsize=(6.4,4.8))

def animate_dots_no_motion(dots_kind_matrix_3D, fig=None, ax=None, mode='subplot'):
    
    if type(dots_kind_matrix_3D) is not list:
        dots_kind_matrix_3D = [dots_kind_matrix_3D]
    
    dots_kind_matrix = dots_kind_matrix_3D[0]
    
    size = 172 * 1 # Set scatter size. 

    # if fig is None:
    #     fig = plt.figure(figsize=(6.4,4.8))
        
    num_vertical, num_horizontal = eg.get_base_dots_info(dots_kind_matrix)
    x_mesh, y_mesh = np.meshgrid(range(num_horizontal), range(num_vertical))

    def scat_dots(ax, dots_kind_matrix, title_str=None):
        ax.axis([-1, num_horizontal, -1, num_vertical]) # Set axis limit
        ax.set_aspect(1) # Normalize the length in the figure
        if not(title_str is None):
            ax.set_title(title_str, fontsize=20)
        
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
        fig, ax, container, = scat_dots_multi_subplot( \
                            fig=fig, \
                            ax=ax, \
                            dots_kind_matrix_3D=dots_kind_matrix_3D, \
                            scat_dots=scat_dots )
        return fig, ax, container,
    else:
        ax = fig.add_subplot(1,1,1)
        if mode == 'anime:func':
            anime = anime_funcUpdate(fig, ax, dots_kind_matrix_3D, scat_dots)
        elif mode == 'anime:artists':
            anime = anime_artists(fig, ax, dots_kind_matrix_3D, scat_dots)
        return fig, ax, anime, 

def scat_dots_multi_subplot(fig, ax, dots_kind_matrix_3D, scat_dots):
    subplot_num = len(dots_kind_matrix_3D)
    subplot_col_num = 4
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
            
        container.append(scat_dots(current_ax, dots_kind_matrix_3D[frame_index], title_str))
        
    return fig, ax, container, 
    
def anime_funcUpdate(fig, ax, dots_kind_matrix_3D, scat_dots):
    if fig is None:
        fig = generate_default_figure()

    def update_frame(frame_index):
        ax.cla()
        title_str = "frame_index = "+str(frame_index)
        scat_dots(ax, dots_kind_matrix_3D[frame_index], title_str)
    
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
    if fig is None:
        fig = generate_default_figure()

    artists=[]
    for frame_index in range( len(dots_kind_matrix_3D) ):
        container = scat_dots(ax, dots_kind_matrix_3D[frame_index])
        title = ax.text(ax.get_xlim()[0],ax.get_ylim()[1]*1.05,"frame_index = "+str(frame_index))
        container.append(title)
        artists.append(container)

    anime = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000) # anime is needed to keep animation visually.
    return anime

# def start_GUI_play():
    
#     import time
#     import tkinter as tk
#     from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    
#     class Application(tk.Frame):
#         def __init__(self, master=None, \
#                 dots_kind_matrix=None, \
#                 num_horizontal = eg.num_horizontal_default, \
#                 num_vertical = eg.num_vertical_default, \
#                 num_kind = eg.num_kind_default, \
#                 num_dummy_kind = eg.num_dummy_kind_default):
            
#             super().__init__(master)
#             self.master = master
#             self.master.title('')
    
#             frame = tk.Frame(self.master) # Frame for matplotlib
#             self.fig = generate_default_figure() # Generate figure for GUI
#             self.ax = self.fig.add_subplot(1, 1, 1) # Generate ax for GUI
#             self.fig_canvas = FigureCanvasTkAgg(self.fig, frame) # Bind frame and matplotlib
#             self.toolbar = NavigationToolbar2Tk(self.fig_canvas, frame) # Set matplotlib toolbar
#             self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True) # Put canvas on the frame
#             frame.pack() # Put frame on the window
            
#             button = tk.Button(self.master, text = "Restart", command = self.initialize) # Set button
#             button.pack(side = tk.BOTTOM) # Put button on the window
            
#             button = tk.Button(self.master, text = "Drop", command = self.button_click) # Set button
#             button.pack(side = tk.BOTTOM) # Put button on the window
            
#             button = tk.Button(self.master, text = "Undo", command = self.button_click) # Set button
#             button.pack(side = tk.BOTTOM) # Put button on the window
            
#             self.master.bind("<KeyPress>", self.key_handler) # Set handler for keyinput
            

#             self.num_horizontal = num_horizontal
#             self.num_vertical = num_vertical
#             self.num_kind = num_kind
#             self.num_dummy_kind = num_dummy_kind
            
#             self.initialize(dots_kind_matrix)
    
#         def button_click(self):
#             # When "Drop" is pressed
#             print(1)
        
#         def key_handler(self, key):
#             print(key.keysym)
#             if key.keysym == "q":
#                 self.master.destroy()
                
#             elif key.keysym == "Right":
#                 self.move_candidate(direction=1)
                
#             elif key.keysym == "Left":
#                 self.move_candidate(direction=-1)
                
#             elif key.keysym == "Up":
#                 self.rot_candidate(direction=1)
                
#             elif key.keysym == "Down":
#                 self.rot_candidate(direction=-1)
                
#             elif key.keysym == "Return":
#                 self.drop_candidate()
#                 self.initiate_candidate()
                
#             elif key.keysym == "r":
#                 self.initiate_candidate()
        
#         def initialize(self, dots_kind_matrix=None):
            
#             self.dots_kind_matrix = dots_kind_matrix
            
#             if self.dots_kind_matrix is None:
#                 # Initialized by an empty box
#                 self.dots_kind_matrix = np.full( (self.num_vertical, self.num_horizontal), -1 )
                
#                 # Randomly generate the initial state
#                 self.dots_kind_matrix = eg.generate_random_dots(num_kind = self.num_kind)
#                 dots_transition, loop_num = \
#                     eg.delete_and_fall_dots_to_the_end(dots_kind_matrix=self.dots_kind_matrix, connected_threshold=4)
#                 self.dots_kind_matrix = dots_transition[-1]
            
#             self.initiate_candidate()
#             self.disp_candidate()
            
#         def initiate_candidate(self):
#             self.next_2dots = np.random.randint(0,self.num_kind+1,(2,1))
#             self.horizontal_index = 0
#             self.is_next_2dots_vertical = True
            
#             self.disp_candidate()
            
#         def move_candidate(self, direction):
#             self.horizontal_index += direction
#             self.disp_candidate()
            
#         def rot_candidate(self, direction):
#             if self.is_next_2dots_vertical:
#                 None
#                 if direction > 0:   # case 0: lower goes left, upper goes right
#                     # Note that this is the inverse of case 3
#                     None
#                 else:               # case 1: lower goes right, upper goes left
#                     # Note that this is the inverse of case 2
#                     self.next_2dots = np.flip(self.next_2dots)
#                     None
#             else:
#                 None
#                 if direction > 0:   # case 2: left goes upper, right goes lower
#                     # Note that this is the inverse of case 1
#                     self.next_2dots = np.flip(self.next_2dots)
#                     None
#                 else:               # case 3: left goes lower, right goes upper
#                     # Note that this is the inverse of case 0
#                     None
#             self.is_next_2dots_vertical = not(self.is_next_2dots_vertical)
#             self.disp_candidate()
            
#         def swap_candidate(self):
#             self.next_2dots = np.flip(self.next_2dots)
#             self.disp_candidate()
            
#         def drop_candidate(self):
#             self.dots_kind_matrix = self.dots_kind_matrix_candidate
#             dots_transition, loop_num = eg.delete_and_fall_dots_to_the_end(self.dots_kind_matrix, 4)
#             for frame_index in range(len(dots_transition)):
#                 self.refresh_scatter_dots(dots_kind_matrix=dots_transition[frame_index])
#                 self.fig_canvas.draw()
#                 self.master.update()
#                 time.sleep(0.5)
#             self.ax.set_title("previos score: "+str(loop_num))
#             self.fig_canvas.draw()
#             self.master.update()
#             self.dots_kind_matrix = dots_transition[-1]
        
#         def disp_candidate(self):
        
#             if self.is_next_2dots_vertical:
#                 if self.horizontal_index == -1:
#                     # When vertical and the most left, move right
#                     self.horizontal_index = 0
#                 elif self.horizontal_index == self.num_horizontal:
#                     # When vertical and the most right, move left
#                     self.horizontal_index = self.num_horizontal - 1
#             else:
#                 if self.horizontal_index == -1:
#                     # When horizontal and the most left, move right
#                     self.horizontal_index = 0
#                 elif self.horizontal_index == self.num_horizontal-1:
#                     # When horizontal and the most right, move left
#                     self.horizontal_index = self.num_horizontal - 2
                
#             # if (not(self.is_next_2dots_vertical)) and (self.horizontal_index == self.num_horizontal):
#             #     # When horizontal and the most right, move left
#             #     self.horizontal_index = self.num_horizontal - 1
        
#             self.dots_kind_matrix_candidate = np.copy(self.dots_kind_matrix)
#             if self.is_next_2dots_vertical:
#                 self.dots_kind_matrix_candidate[-2,self.horizontal_index] = self.next_2dots[0]
#                 self.dots_kind_matrix_candidate[-1,self.horizontal_index] = self.next_2dots[1]
#             else:
#                 self.dots_kind_matrix_candidate[-2,self.horizontal_index] = self.next_2dots[0]
#                 self.dots_kind_matrix_candidate[-2,self.horizontal_index+1] = self.next_2dots[1]
            
#             self.refresh_scatter_dots(self.dots_kind_matrix_candidate)
        
#         def refresh_scatter_dots(self, dots_kind_matrix):
#             title_str = self.ax.get_title()
#             self.ax.cla()
#             animate_dots_no_motion(dots_kind_matrix_3D=dots_kind_matrix, fig=self.fig, ax=self.ax)
#             self.ax.set_title(title_str)
#             self.fig_canvas.draw()
            
            
    
#     root = tk.Tk()
#     app = Application(master=root)
#     app.mainloop()
        