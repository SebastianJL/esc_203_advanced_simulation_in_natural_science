"""Solve the 2D laplace equation, i.e. grad^2 u(x,y) = 0.
Discretization on a grid: Finite difference for laplace operator.
Use of over-relaxation for faster solution:
Model problem: Voltage across a grid.
Boundary conditions:
- 0V at outer boundaries.
- 1000V on a line in the middle of the grid.

Laplace equation:
grad^2 phi(x,y) = 0

Poisson equation:
grad^2 phi(x,y) = rho(x,y)

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def main():
    max_restrictions = 7
    n_restrictions = 4
    dx = 1
    assert n_restrictions <= max_restrictions
    # Needs to be 2^n + 1.
    n_grid = 2**max_restrictions + 1

    # Voltage grid.
    voltage_grid = np.zeros((n_grid, n_grid))
    high_voltage_index = (n_grid//2 + 1, slice(10, -10))
    high_voltage_mask = np.zeros_like(voltage_grid, dtype=bool)
    high_voltage_mask[high_voltage_index] = True
    voltage_grid[high_voltage_mask] = 1000

    # Pre smoothing. SOR replaces voltage_grid in place.
    iter_count = sor(voltage_grid, dx, high_voltage_mask, w=1, max_iterations=10)

    defect = calculate_defect(voltage_grid, rho=0, boundary_mask=high_voltage_mask)

    # Restriction.
    for _ in range(n_restrictions):
        defect, high_voltage_mask = restrict(defect, high_voltage_mask)
        defect[high_voltage_mask] = 0

    # Solve problem on coarse grid.
    correction = np.zeros_like(defect)
    high_voltage_mask = np.zeros_like(defect, dtype=bool)

    iter_count = sor(correction, dx, high_voltage_mask, w=1, rho=defect, max_iterations=1000)

    # Prolongation.
    for _ in range(n_restrictions):
        correction = prolongate(correction)

    voltage_grid += correction

    # print(f'{w = }')
    # print(f'{t.elapse = }s')
    print(f'{iter_count = }')
    print(f'{voltage_grid.max() = }')

    levels = np.linspace(0, 1000, 11)
    plt.contourf(voltage_grid, levels=levels)
    plt.colorbar()
    plt.title(f'{iter_count = }')
    plt.savefig('laplace_multigrid.png')


def calculate_defect(grid, rho, boundary_mask):
    """Calculate the defect of the voltage grid for the laplace equation.

    Laplace: L_continuous*phi_exact = rho_continuous
    Defect := rho_discrete - L_discrete*phi_estimate
    where L = grad^2
    """

    poisson_kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])

    defect = rho - ndimage.correlate(grid, poisson_kernel, mode='constant', cval=0)
    # Set defect at boundary to zero, because there is no defect at the boundary.
    defect[boundary_mask] = 0

    return defect


def sor(grid, dx, high_voltage_mask, w=1, rho=0, max_iterations=1000) -> int:
    """Successive over-relaxation solver.

    Grid is changed inplace.

    For w=1 this corresponds to Gauss-Seidel.
    Returns the number of iterations.
    """

    # Stencil for 2D laplace finite differences method.
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    residual = np.full_like(grid, np.inf)

    omega = np.full_like(residual, w/4)
    # We set the omega to 0 where the boundary conditions apply. I.e. no update there. Boundary condition at outer
    # border of grid are ensured by correlation mode = 'constant'.
    omega[high_voltage_mask] = 0

    # Construct checkerboard pattern.
    white_tiles = np.ones_like(grid, dtype=bool)
    white_tiles[::2, ::2] = False
    white_tiles[1::2, 1::2] = False
    black_tiles = ~white_tiles
    iter_count = 0

    while residual.max() > 1.0 and iter_count < max_iterations:
        ndimage.correlate(grid, kernel, output=residual, mode='constant', cval=0)
        grid[black_tiles] += (omega*(residual - dx**2*rho))[black_tiles]

        ndimage.correlate(grid, kernel, output=residual, mode='constant', cval=0)
        grid[white_tiles] += (omega*(residual - dx**2*rho))[white_tiles]
        iter_count += 1

    return iter_count


def restrict(grid, boundary_mask):

    kernel = 1/4 * np.array([
        [1/4, 1/2, 1/4],
        [1/2, 1/1, 1/2],
        [1/4, 1/2, 1/4],
    ])

    restricted_grid = ndimage.correlate(grid, kernel, mode='constant', cval=0)
    coarse_grid = restricted_grid[::2, ::2]
    caorse_boundary_mask = boundary_mask[::2, ::2]

    return coarse_grid, caorse_boundary_mask


def prolongate(grid):

    kernel = np.array([
        [1/4, 1/2, 1/4],
        [1/2, 1/1, 1/2],
        [1/4, 1/2, 1/4],
    ])

    fine_grid = np.zeros((grid.shape[0]*2 - 1, grid.shape[1]*2 - 1))
    fine_grid[::2, ::2] = grid
    fine_grid = ndimage.correlate(fine_grid, kernel, mode='constant', cval=0)

    return fine_grid


if __name__ == '__main__':
    main()
