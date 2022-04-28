# Implementing Multi-grid method for poisson equation
# Solve poisson equation using SOR method given some boundary conditions
# Create a contour plot of the resulting potential

# Poisson's equation: nabla^2 V = - phi_v / epsilon

# Imports
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import time
from copy import deepcopy
from math import floor, ceil
from numba import jit


def grid_rand(length, height, dx, dy):
    """Returns a grid of given size
    length and height should be a multiple of their respective step size dx and dy"""
    l_grid_steps = int(length/dx)
    h_grid_steps = int(height/dy)
    grid = np.array([[np.random.randint(0, 9) for i in range(h_grid_steps)] for j in range(l_grid_steps)])
    c_grid = np.ones(grid.shape, dtype=bool)
    return grid, c_grid


def grid_init(length, height, dx, dy):
    """Returns a grid of given size
    length and height should be a multiple of their respective step size dx and dy"""
    l_girid_steps = int(length/dx)
    h_grid_steps = int(height/dy)
    grid = np.zeros((h_grid_steps, l_girid_steps))
    c_grid = np.ones(grid.shape, dtype=bool)
    return grid, c_grid


def set_plate(grid, x1, y1, x2, y2, potential, c_grid):
    length, height = grid.shape
    x1_grid, y1_grid, x2_grid, y2_grid = int(x1 * length), int(y1 * height), int(x2 * length), int(y2 * height)

    n = ceil(np.sqrt((x2_grid - x1_grid)**2 + (y2_grid - y1_grid)**2)) + 1
    x = np.linspace(x1_grid, x2_grid, n)
    x = np.array([floor(i) for i in x])
    y = np.linspace(y1_grid, y2_grid, n)
    y = np.array([floor(i) for i in y])

    # Conditional grid init
    cond_grid = c_grid

    for i, x_i in enumerate(x):
        grid[x_i, y[i]] = potential
        cond_grid[x_i, y[i]] = False

    return grid, cond_grid


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


def main_loop(grid, boundries, max_steps, w):
    for i in range(max_steps):
        old_grid = deepcopy(grid)
        new_grid = sweep(grid, boundries, w)
        if abs(np.sum(old_grid) - np.sum(new_grid)) < 0.001:
            print(f"Converged in {i} steps")
            return new_grid

        grid = new_grid

    print("Reached max steps")
    return new_grid


def plot_grid(grid):
    plt.figure()
    levels = np.linspace(0, 1000, 11)
    cont = plt.contourf(grid, levels=levels)
    cbar = plt.colorbar(cont)
    plt.show()


def restriction(grid, boundries):
    stencil = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1/4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])

    coarse_grid = ndimage.convolve(grid, stencil, mode="constant", cval=0)

    coarse_grid = coarse_grid[1::2, 1::2]

    # adjust boundaries to same size
    new_boundries = boundries[1::2, 1::2]

    return coarse_grid, new_boundries


def prolongation(grid, boundries):
    stencil = np.array([[1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4]])

    grid_new = np.repeat(grid, 2, axis=0)
    grid_new = np.repeat(grid_new, 2, axis=1)
    # instead of repeating make every second element 0
    grid_new[::2, :] = 0
    grid_new[:, ::2] = 0

    fine_grid = ndimage.convolve(grid_new, stencil, mode="constant", cval=0)

    # adjust boundries to same size
    new_boundries = np.repeat(boundries, 2, axis=0)
    new_boundries = np.repeat(new_boundries, 2, axis=1)

    return fine_grid, new_boundries


def test_01():
    grid, cond_grid = grid_rand(10, 10 , 1, 1)

    print(grid)

    print("restriciton")
    grid = restriction(grid, cond_grid)
    print(grid)

    print("prolongation")
    grid = prolongation(grid, cond_grid)
    print(grid)

    print("restriciton2")
    grid = restriction(grid, cond_grid)
    print(grid)

    print("prolongation2")
    grid = prolongation(grid, cond_grid)
    print(grid)


def multigrid_solve(depth, grid, boundries, max_steps, w, plate_x1, plate_y1, plate_x2, plate_y2, potential):
    new_grid = grid
    #plot_grid(new_grid)
    # Restrictions
    for i in range(depth):
        new_grid, boundries = restriction(new_grid, boundries)
        new_grid, boundries = set_plate(new_grid, plate_x1, plate_y1, plate_x2, plate_y2, potential, boundries)
        #plot_grid(new_grid)
    # Solve
    new_grid = main_loop(new_grid, boundries, max_steps, w)
    #plot_grid(new_grid)
    # Prolongation
    for i in range(depth):
        new_grid, boundries = prolongation(new_grid, boundries)
        new_grid, boundries = set_plate(new_grid, 0.51, 0.2, 0.51, 0.8, 1000, boundries)
        #plot_grid(new_grid)

    return new_grid


w = 1.3      # w = 1 is Jacobian, w > 1 is SOR
steps = 15_000
depth = 1
grid1, cond_grid = grid_init(200, 200, 1, 1)

plate_x1, plate_y1, plate_x2, plate_y2, potential = 0.5, 0.2, 0.5, 0.8, 1000
grid1, boundaries = set_plate(grid1, plate_x1, plate_y1, plate_x2, plate_y2, potential, cond_grid)

t0 = time.time()
out_grid = multigrid_solve(depth, grid1, boundaries, steps, w, plate_x1, plate_y1, plate_x2, plate_y2, potential)
t1 = time.time()
print(f"Calculation time: {t1-t0}s")


# Plot


plt.figure()
plt.title("Potential in a box with 1000 V plate")
levels = np.linspace(0, 1000, 11)
cont = plt.contourf(out_grid, levels=levels)
cbar = plt.colorbar(cont)
cbar.ax.set_title("Potential [V]", fontsize=12, loc="left")
plt.savefig("multigrid_plot.png")


#TODO: restruction and prolongation functions need to be adjusted for boundary conditions (different convolve at edges?)
# or make a control function for boundries to adjust them after the restriction/prolongation