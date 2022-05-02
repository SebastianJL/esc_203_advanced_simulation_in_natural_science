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
    max_restrictions = 9
    n_restrictions = 3
    assert n_restrictions <= max_restrictions
    # Needs to be 2^n + 1.
    n_grid = 2**max_restrictions + 1
    dx = 1/n_grid
    print(f'{n_grid}')

    # Voltage grid.
    voltage_grid = np.zeros((n_grid, n_grid))
    high_voltage_mask = np.zeros_like(voltage_grid, dtype=bool)
    high_voltage_mask[n_grid//2, 10:-10] = True
    voltage_grid[high_voltage_mask] = 1000

    # Pre smoothing. SOR replaces voltage_grid in place.
    iter_count = jacobi(voltage_grid, dx, 0, high_voltage_mask, max_iterations=10)

    residual = calculate_residual(voltage_grid, dx, rho=0, boundary_mask=high_voltage_mask)

    # Restriction.
    high_voltage_masks = [high_voltage_mask]
    for _ in range(n_restrictions):
        residual, high_voltage_mask = restrict(residual, high_voltage_mask)
        residual[high_voltage_mask] = 0
        high_voltage_masks.append(high_voltage_mask)
        dx *= 2

    # Solve residual equation on coarse grid.
    error = np.zeros_like(residual)
    v_cycle_iterations = jacobi(error, dx, rho=residual, boundary_mask=high_voltage_mask, max_iterations=1000)
    print(f'{v_cycle_iterations = }')

    # Prolongation.
    for i in range(2, n_restrictions + 2):
        error = prolongate(error)
        dx /= 2
        error[high_voltage_masks[-i]] = 0

    voltage_grid += error

    iter_count = jacobi(voltage_grid, dx, rho=0, boundary_mask=high_voltage_masks[0], max_iterations=1000)

    # print(f'{w = }')
    # print(f'{t.elapse = }s')
    print(f'{iter_count = }')
    print(f'{voltage_grid.max() = }')

    levels = np.linspace(0, 1000, 11)
    plt.contourf(voltage_grid, levels=levels)
    plt.colorbar()
    plt.title(f'{iter_count = }')
    plt.savefig('laplace_multigrid.png')


def calculate_residual(grid, dx, rho, boundary_mask):
    """Calculate the defect of the grid for the laplace equation.

    Laplace: L_continuous*phi_exact = rho_continuous
    Defect := rho_discrete - L_discrete*phi_estimate
    where L = grad^2
    """

    poisson_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    residual = rho - 1/dx**2*ndimage.convolve(grid, poisson_kernel, mode='constant', cval=0)
    # Set defect at boundary to zero, because there is no defect at the boundary.
    residual[boundary_mask] = 0

    return residual


def jacobi(grid, dx, rho, boundary_mask, max_iterations=1000) -> int:
    """Iterative Jacobi solver.

    Solves the discrete Poisson equation:
    L * grid = rho
    where L = grad^2.

    Grid is changed inplace.

    :param grid: 2D array.
    :param dx: Grid spacing.
    :param rho: 2D array with same dimensions as grid.
    :param boundary_mask: grid[boundary_mask] will stay untouched.
    :param max_iterations: Maximum number of iterations.
    :return: The number of iterations.
    """

    poisson_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    # Jacobi iteration matrix.
    correction = np.full_like(grid, np.inf)

    iter_count = 0
    while abs(correction.max()) > 1 and iter_count < max_iterations:
        ndimage.correlate(grid, poisson_kernel, output=correction, mode='constant', cval=0)
        correction -= dx**2*rho
        grid[~boundary_mask] += (1/4*correction)[~boundary_mask]
        iter_count += 1

    return iter_count


def restrict(grid, boundary_mask):
    kernel = 1/4*np.array([
        [1/4, 1/2, 1/4],
        [1/2, 1/1, 1/2],
        [1/4, 1/2, 1/4],
    ])

    restricted_grid = ndimage.correlate(grid, kernel, mode='constant', cval=0)
    coarse_grid = restricted_grid[::2, ::2]
    coarse_boundary_mask = boundary_mask[::2, ::2]

    return coarse_grid, coarse_boundary_mask


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
