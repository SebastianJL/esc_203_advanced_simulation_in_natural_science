"""Solve the 2D laplace equation, i.e. grad^2 u(x,y) = 0.
Discretization on a grid: Finite difference for laplace operator.
Use of over-relaxation for faster solution:
U_new = w/4 * R - U_old
Model problem: Voltage across a grid.
Boundary conditions:
- 0V at outer boundaries.
- 1000V on a line in the middle of the grid.
"""
import matplotlib.pyplot as plt
import numba
import numpy as np
from timer import timer


@numba.njit()
def correlate(u, *, mask, out):
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            if not mask[i, j]:
                continue
            out[i, j] = 0
            if i > 0:
                out[i, j] += u[i - 1, j]
            if i < u.shape[0] - 1:
                out[i, j] += u[i + 1, j]
            if j > 0:
                out[i, j] += u[i, j - 1]
            if j < u.shape[1] - 1:
                out[i, j] += u[i, j + 1]
            out[i, j] -= 4*u[i, j]


def main():
    n_grid = 100

    # Voltage grid.
    u = np.zeros((n_grid, n_grid))
    high_voltage_mask = (n_grid//2, slice(10, -10))
    u[high_voltage_mask] = 1000

    # Residual grid.
    residual = np.full_like(u, 10)

    # Stencil for 2D laplace finite differences method.
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    # Relaxation constant. Approximation formula taken from a lecture script.
    w = 2/(1 + np.pi/n_grid)
    omega = np.full_like(residual, w/4)
    # We set the omega to 0 where the boundary conditions apply. I.e. no update there. Boundary condition at outer
    # border of grid are ensured by convolution mode = 'constant'.
    omega[high_voltage_mask] = 0

    # Construct checkerboard pattern.
    white_tiles = np.ones_like(u, dtype=bool)
    white_tiles[::2, ::2] = False
    white_tiles[1::2, 1::2] = False
    black_tiles = ~white_tiles

    iter_count = 0
    max_iterations = 1_000

    with timer() as t:
        while residual.max() > 1.0 and iter_count < max_iterations:
            # Correlation
            correlate(u, mask=black_tiles, out=residual)
            u[black_tiles] += (omega*residual)[black_tiles]

            correlate(u, mask=white_tiles, out=residual)
            u[white_tiles] += (omega*residual)[white_tiles]

            iter_count += 1

    print(f'{w = }')
    print(f'{t.elapse = }s')
    print(f'{iter_count = }')
    print(f'{residual.max() = }')

    levels = np.linspace(0, 1000, 11)
    plt.contourf(u, levels=levels)
    plt.colorbar()
    plt.title(f'{iter_count = }, {w = }')
    plt.savefig('laplace.png')


if __name__ == '__main__':
    main()
