# Solve poisson equation using SOR method given some boundary conditions
# Create a contour plot of the resulting potential

# Poisson's equation: nabla^2 V = - phi_v / epsilon

# Imports
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import time
from numba import jit

def grid_init(length, height, dx, dy):
    """Returns a grid of given size
    length and height should be a multiple of their respective step size dx and dy"""
    l_girid_steps = int(length/dx)
    h_grid_steps = int(height/dy)
    grid = np.zeros((h_grid_steps, l_girid_steps))
    c_grid = np.ones(grid.shape, dtype=bool)
    return grid, c_grid

def set_plate(grid, x1, y1, x2, y2, potential, c_grid):
    """Place a plate from x1/y2 to x2/y2 in grid and set its potential
    x and y given as a fraction of the length/height of the grid"""
    length, height = grid.shape
    x1_grid, y1_grid, x2_grid, y2_grid = int(x1 * length), int(y1 * height), int(x2 * length), int(y2 * height)
    # y = m*x + c
    dx = x2_grid - x1_grid
    dy = y2_grid - y1_grid
    if dx == 0:
        m = 0
    else:
        m = dy/dx

    c = y1_grid - m * x1_grid
    # Conditional grid init
    cond_grid = c_grid
    # Set potential
    for i in range(length):
        if max(x1_grid, x2_grid) >= i >= min(x1_grid, x2_grid):
            for j in range(height):
                if j == int(round(m * i + c)) and max(y1_grid, y2_grid) >= j >= min(y1_grid, y2_grid):
                    grid[j, i] = potential
                    cond_grid[j, i] = False
    return grid, cond_grid

#TODO: rewrite the set_plate function so it works consistantly

# Stencil sweep
def sweep(grid, boundries, w):
    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    grid_out = np.zeros_like(grid)
    # Sweep 2 times in checkerboard pattern (black then white)
    # Conditional grid
    C = np.zeros(grid.shape, dtype=bool)
    C[::2, ::2] = True
    C[1::2, 1::2] = True
    boundries_black = C * boundries
    C_bar = np.ones(grid.shape, dtype=bool)
    C_bar[::2, ::2] = False
    C_bar[1::2, 1::2] = False
    boundries_white = C_bar * boundries

    # sweep 1
    grid[boundries_black] += (w/4) * ndimage.convolve(grid, stencil, output=grid_out, mode="constant", cval=0)[boundries_black]
    # sweep 2
    grid[boundries_white] += (w/4) * ndimage.convolve(grid, stencil, output=grid_out, mode="constant", cval=0)[boundries_white]

    return grid

def main_loop(grid, boundries, steps, w):
    for i in range(steps):
        grid = sweep(grid, boundries, w)
    return grid


if __name__ == '__main__':

    #TODO: in main loop run sweep until some condition of minimal change is met

    w = 1.3     # w = 1 is Jacobian, w > 1 is SOR
    grid1, cond_grid = grid_init(100, 100, 1, 1)

    grid1, boundaries = set_plate(grid1, 0.1, 0.5, 0.9, 0.5, 1000, cond_grid)

    steps = 5_000
    t0 = time.time()
    out_grid = main_loop(grid1, boundaries, steps, w)
    t1 = time.time()
    print(f"Calculation time: {t1-t0}s")


    # Plot


    plt.figure()
    plt.title("Potential in a box with 1000 V plate")
    levels = np.linspace(0, 1000, 11)
    cont = plt.contourf(out_grid, levels=levels)
    cbar = plt.colorbar(cont)
    cbar.ax.set_title("Potential [V]", fontsize=12, loc="left")
    plt.savefig("Potential_contour.png")
