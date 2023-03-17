import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import UI_dots as ui
from functions import engine_dots as eg

def GUI_to_play():
    
    import time
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    
    class Application(tk.Frame):
        def __init__(self, master=None, \
                dots_kind_matrix=None, \
                num_horizontal = eg.num_horizontal_default, \
                num_vertical = eg.num_vertical_default, \
                num_kind = eg.num_kind_default, \
                num_dummy_kind = eg.num_dummy_kind_default):
            
            super().__init__(master)
            self.master = master
            self.master.title('')
    
            frame = tk.Frame(self.master) # Frame for matplotlib
            self.fig = ui.generate_default_figure() # Generate figure for GUI
            self.ax = self.fig.add_subplot(1, 1, 1) # Generate ax for GUI
            self.fig_canvas = FigureCanvasTkAgg(self.fig, frame) # Bind frame and matplotlib
            self.toolbar = NavigationToolbar2Tk(self.fig_canvas, frame) # Set matplotlib toolbar
            self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True) # Put canvas on the frame
            frame.pack() # Put frame on the window
            
            button = tk.Button(self.master, text = "Restart", command = self.initialize) # Set button
            button.pack(side = tk.BOTTOM) # Put button on the window
            
            button = tk.Button(self.master, text = "Drop", command = self.Drop) # Set button
            button.pack(side = tk.BOTTOM) # Put button on the window
            
            button = tk.Button(self.master, text = "Undo", command = self.Undo) # Set button
            button.pack(side = tk.BOTTOM) # Put button on the window
            
            self.master.bind("<KeyPress>", self.key_handler) # Set handler for keyinput
            

            self.num_horizontal = num_horizontal
            self.num_vertical = num_vertical
            self.num_kind = num_kind
            self.num_dummy_kind = num_dummy_kind
            
            self.initialize(dots_kind_matrix)
    
        def Drop(self):
            self.is_falling = True
            self.master.update()
            self.drop_candidate()
            self.initiate_candidate()
    
        def Undo(self):
            self.dots_kind_matrix = self.dots_kind_matrix_previous
            self.next_2dots = self.next_2dots_previous
            self.initiate_candidate(is_undo=True)
        
        def key_handler(self, key):
            print(key.keysym)
            if not(self.is_dropping):
                if key.keysym == "q":
                    self.master.destroy()
                    
                elif key.keysym == "Right":
                    self.move_candidate(direction=1)
                    
                elif key.keysym == "Left":
                    self.move_candidate(direction=-1)
                    
                elif key.keysym == "Up":
                    self.rot_candidate(direction=1)
                    
                elif key.keysym == "Down":
                    self.rot_candidate(direction=-1)
                    
                elif key.keysym == "Return":
                    self.Drop()
                    
                elif key.keysym == "r":
                    self.initiate_candidate()
                    
                elif key.keysym == "u":
                    self.Undo()
        
        def initialize(self, dots_kind_matrix=None):
            
            self.dots_kind_matrix = dots_kind_matrix
            
            if self.dots_kind_matrix is None:
                # Initialized by an empty box
                self.dots_kind_matrix = np.full( (self.num_vertical, self.num_horizontal), 0 )
                
                # # Randomly generate the initial state
                # self.dots_kind_matrix = eg.generate_random_dots(num_kind = self.num_kind)
                # dots_transition, loop_num = \
                #     eg.delete_and_fall_dots_to_the_end(dots_kind_matrix=self.dots_kind_matrix, connected_threshold=4)
                # self.dots_kind_matrix = dots_transition[-1]
            
            self.initiate_candidate()
            self.disp_candidate()
            
            self.dots_kind_matrix_previous = self.dots_kind_matrix
            self.next_2dots_previous = self.next_2dots
            self.is_dropping = False
            
        def initiate_candidate(self,is_undo=False):
            if is_undo:
                self.next_2dots = self.next_2dots_previous
            else:
                # default
                self.next_2dots = np.random.randint(1,self.num_kind+1,(2,1))
                
            self.horizontal_index = 0
            self.is_next_2dots_vertical = True
            
            self.disp_candidate()
            
        def move_candidate(self, direction):
            self.horizontal_index += direction
            self.disp_candidate()
            
        def rot_candidate(self, direction):
            if self.is_next_2dots_vertical:
                None
                if direction > 0:   # case 0: lower goes left, upper goes right
                    # Note that this is the inverse of case 3
                    None
                else:               # case 1: lower goes right, upper goes left
                    # Note that this is the inverse of case 2
                    self.next_2dots = np.flip(self.next_2dots)
                    None
            else:
                None
                if direction > 0:   # case 2: left goes upper, right goes lower
                    # Note that this is the inverse of case 1
                    self.next_2dots = np.flip(self.next_2dots)
                    None
                else:               # case 3: left goes lower, right goes upper
                    # Note that this is the inverse of case 0
                    None
            self.is_next_2dots_vertical = not(self.is_next_2dots_vertical)
            self.disp_candidate()
            
        def swap_candidate(self):
            self.next_2dots = np.flip(self.next_2dots)
            self.disp_candidate()
            
        def drop_candidate(self):
            
            self.is_dropping = True
            
            self.dots_kind_matrix_previous = self.dots_kind_matrix
            self.next_2dots_previous = self.next_2dots
            
            self.dots_kind_matrix = self.dots_kind_matrix_candidate
            dots_transition, loop_num = eg.delete_and_fall_dots_to_the_end(self.dots_kind_matrix, 4)
            
            for frame_index in range(len(dots_transition)):
                if not(frame_index == 0):
                    time.sleep(0.1)
                self.refresh_scatter_dots(dots_kind_matrix=dots_transition[frame_index])
                self.fig_canvas.draw()
                self.master.update()
            
            self.ax.set_title("previos score: "+str(loop_num))
            self.fig_canvas.draw()
            self.master.update()
            self.dots_kind_matrix = dots_transition[-1]
            
            self.is_dropping = False
        
        def disp_candidate(self):
        
            if self.is_next_2dots_vertical:
                if self.horizontal_index == -1:
                    # When vertical and the most left, move right
                    self.horizontal_index = 0
                elif self.horizontal_index == self.num_horizontal:
                    # When vertical and the most right, move left
                    self.horizontal_index = self.num_horizontal - 1
            else:
                if self.horizontal_index == -1:
                    # When horizontal and the most left, move right
                    self.horizontal_index = 0
                elif self.horizontal_index == self.num_horizontal-1:
                    # When horizontal and the most right, move left
                    self.horizontal_index = self.num_horizontal - 2
                
            # if (not(self.is_next_2dots_vertical)) and (self.horizontal_index == self.num_horizontal):
            #     # When horizontal and the most right, move left
            #     self.horizontal_index = self.num_horizontal - 1
        
            self.dots_kind_matrix_candidate = np.copy(self.dots_kind_matrix)
            if self.is_next_2dots_vertical:
                self.dots_kind_matrix_candidate[-2,self.horizontal_index] = self.next_2dots[0]
                self.dots_kind_matrix_candidate[-1,self.horizontal_index] = self.next_2dots[1]
            else:
                self.dots_kind_matrix_candidate[-2,self.horizontal_index] = self.next_2dots[0]
                self.dots_kind_matrix_candidate[-2,self.horizontal_index+1] = self.next_2dots[1]
            
            self.refresh_scatter_dots(dots_kind_matrix=self.dots_kind_matrix_candidate)
        
        def refresh_scatter_dots(self, dots_kind_matrix=None):
            if dots_kind_matrix is None:
                dots_kind_matrix = self.dots_kind_matrix
                
            title_str = self.ax.get_title()
            self.ax.cla()
            ui.animate_dots_no_motion(dots_kind_matrix_3D=dots_kind_matrix, fig=self.fig, ax=self.ax)
            self.ax.set_title(title_str)
            self.fig_canvas.draw()
            
            
    
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()


GUI_to_play()