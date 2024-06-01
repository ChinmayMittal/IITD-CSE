import numpy as np
import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_traj_frames(grid, policy,max_frames = 100):
    # start state will be coded as red, free state as white, goal state as green, hole state as black
    frames = []
    total = 0
    curr_grid = np.zeros_like(grid, dtype = np.uint8)
    curr_grid[grid == 'S'] = 0
    curr_grid[grid == 'F'] = 100
    curr_grid[grid == 'G'] = 200
    curr_grid[grid == 'H'] = 255
    
    frames.append(curr_grid)
    # now get the frames assuming a deterministic transition model for now
    while True and total < max_frames:
        total+=1
        r, c = np.where(curr_grid == 0)
        r, c = r[0], c[0]
        r_old = r
        c_old = c
    
        action = policy[r][c]
        if action == -1:
            # either you have reached the terminal state or the hole state
            
            break
        if action == 0:
            r = max(r-1, 0)
        elif action == 1:
            r = min(r+1, grid.shape[0]-1)
        elif action == 2:
            c = max(c-1, 0)
        elif action == 3:
            c = min(c+1, grid.shape[1]-1)

        # change the grid
        curr_grid_new = deepcopy(curr_grid)
        curr_grid_new[r_old][c_old] = '100'
        curr_grid_new[r][c] = '0'

        frames.append(curr_grid_new)
        if grid[r][c] == 'G':
            break

        curr_grid = curr_grid_new

    return frames



def update_frame(canvas, frames, frame_index, delay, cell_size = 50):
    if frame_index < len(frames):
        canvas.delete("all")  # Clear the canvas
        draw_grid(canvas, frames[frame_index], cell_size = cell_size)
        
        # Schedule the next frame update
        canvas.after(delay, update_frame, canvas, frames, frame_index + 1, delay, cell_size)
    return 

def draw_grid(canvas, array, cell_size = 50):
    nrows, ncols = array.shape
    cell_size = cell_size
    cmap = plt.get_cmap('viridis')  # Choose a color map
    norm = mcolors.Normalize(vmin=0, vmax=255)
    
    for i in range(nrows):
        for j in range(ncols):
            # Normalize the array value to get a corresponding color from the color map
            normalized_value = norm(array[i, j])
            rgba = cmap(normalized_value)  # Get RGBA value from the color map
            hex_color = mcolors.rgb2hex(rgba)
            canvas.create_rectangle(j*cell_size, i*cell_size, (j+1)*cell_size, (i+1)*cell_size, fill=hex_color, outline='black')

def plot(frames, frame_delay=500, cell_size=50):
    '''
    frames: the list of numpy arrays where each array is a state of the grid
    frame_delay: the delay between consecutive frame in milliseconds while rendering
    cell_size: the size of each cell in grid world
    '''
    root = tk.Tk()
    root.title("2D Array Sequence")

    canvas_height = frames[0].shape[0] * cell_size
    canvas_width = frames[0].shape[1] * cell_size
    canvas = tk.Canvas(root, height=canvas_height, width=canvas_width)
    canvas.pack()

    
    update_frame(canvas, frames, 0, frame_delay, cell_size)
    root.mainloop()
    
