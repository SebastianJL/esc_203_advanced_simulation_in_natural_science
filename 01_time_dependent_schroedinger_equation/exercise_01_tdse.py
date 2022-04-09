"""Numerical solver for a 1D chain of fermions with a generic time independent potential.
I worked together with Tuna, Sujeni and Casper.
"""


import numpy as np
from matplotlib import pyplot as plt, animation
from scipy import sparse


def quantum_tunneling(show_ani: bool = True, save_ani: bool = False):
    # Define parameters
    k = 1  # Wave number
    lambda_ = 1  # Parameter
    sigma = 6*lambda_  # Standard deviation
    delta = 0.1*lambda_  # Grid spacing
    dt = 0.01*lambda_  # Time step
    Lx = 2000  # Grid size
    n_iter = 200_000  # Number of iterations in time.
    assert (Lx%2 == 0)

    # Define wave function phi
    x = np.arange(Lx)*delta
    x0 = 4*sigma
    phi = np.exp(-2*np.pi*1j*k*x)*np.exp(-(x - x0)**2/(2*sigma**2))
    support = (x > x0 - 3*sigma) & (x < x0 + 3*sigma)
    phi[~support] = 0

    # Normalize phi
    norm = np.linalg.norm(phi)
    # norm = np.sqrt(phi.dot(phi.conjugate()))  # Alternate form.
    phi = 1/norm*phi

    # Define potential V
    V = np.zeros_like(phi, dtype=float)
    V[800: 1200] = 4.5

    # Define 2-particle operator M(dt/2)
    alpha = 1/(4*np.pi**2*delta**2)
    alpha_abs = np.abs(alpha)
    M = np.array([
        [np.cos(dt/2*alpha_abs), -1j*np.sin(dt/2*alpha_abs)],
        [-1j*np.sin(dt/2*alpha_abs), np.cos(dt/2*alpha_abs)],
    ])

    # Define U1(dt/2), U2(dt/2) and U3(dt/2)
    U1 = sparse.kron(sparse.identity(Lx//2), M)

    U2 = np.zeros((Lx, Lx), dtype=complex)
    U2[1:-1, 1:-1] = np.kron(np.identity(Lx//2 - 1), M)
    U2 = sparse.dia_array(U2)

    U3 = np.exp(-1j*dt*alpha*(2 + 4*np.pi**2*delta**2*V))
    U3 = sparse.diags(U3)
    assert (U1.shape == U2.shape == U3.shape)

    # Define second order U(dt)=U(dt/2)@U_dagger(dt/2)
    U = U3@U2@U1@U1@U2@U3
    U = sparse.dia_array(U)

    # Evolve phi over time.
    phis = [phi]
    norms = [np.linalg.norm(phi)]
    for i in range(n_iter):
        phi = U@phi
        if i%100 == 0:
            phis.append(phi)
            norms.append(np.linalg.norm(phi))

    # Animate
    def update_line(i, data, norms, line, ax):
        ax.title.set_text(f'frame = {i:03}, 1-norm={1 - norms[i]:.3e}')
        line.set_ydata(np.abs(data[i]))
        return line,

    # Scale V so we can put it in the same plot as phi.
    V_plot = V/np.max(V)*np.max(phis[0].real)*2

    fig, ax = plt.subplots()
    ax.title.set_text(f'frame = {0:03}, 1-norm={1 - norms[0]:.3e}')
    l, = ax.plot(x, np.abs(phis[0]), 'r-')
    ax.plot(x, V_plot)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$|\phi(x)|$')
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$V(x)$')
    line_ani = animation.FuncAnimation(fig, update_line, len(phis), fargs=(phis, norms, l, ax), interval=15,
                                       repeat=True)

    if save_ani:
        line_ani.save('tunneling.mp4')
    if show_ani:
        plt.show()


def parabolic_potential_well(show_ani: bool = True, save_ani: bool = False):
    # Define parameters
    k = 1  # Wave number
    lambda_ = 1  # Parameter
    sigma = 6*lambda_  # Standard deviation
    delta = 0.1*lambda_  # Grid spacing
    dt = 0.01*lambda_  # Time step
    Lx = 2000  # Grid size
    n_iter = 200_000  # Number of iterations in time.
    assert (Lx%2 == 0)

    # Define wave function phi
    x = np.arange(Lx)*delta
    x0 = x[Lx//2]
    phi = np.exp(-2*np.pi*1j*k*x)*np.exp(-(x - x0)**2/(2*sigma**2))
    support = (x > x0 - 3*sigma) & (x < x0 + 3*sigma)
    phi[~support] = 0

    # Normalize phi
    norm = np.linalg.norm(phi)
    # norm = np.sqrt(phi.dot(phi.conjugate()))  # Alternate form.
    phi = 1/norm*phi

    # Define potential V
    V = 0.001*(x - x[Lx//2])**2

    # Define 2-particle operator M(dt/2)
    alpha = 1/(4*np.pi**2*delta**2)
    alpha_abs = np.abs(alpha)
    M = np.array([
        [np.cos(dt/2*alpha_abs), -1j*np.sin(dt/2*alpha_abs)],
        [-1j*np.sin(dt/2*alpha_abs), np.cos(dt/2*alpha_abs)],
    ])

    # Define U1(dt/2), U2(dt/2) and U3(dt/2)
    U1 = sparse.kron(sparse.identity(Lx//2), M)

    U2 = np.zeros((Lx, Lx), dtype=complex)
    U2[1:-1, 1:-1] = np.kron(np.identity(Lx//2 - 1), M)
    U2 = sparse.dia_array(U2)

    U3 = np.exp(-1j*dt*alpha*(2 + 4*np.pi**2*delta**2*V))
    U3 = sparse.diags(U3)
    assert (U1.shape == U2.shape == U3.shape)

    # Define second order U(dt)=U(dt/2)@U_dagger(dt/2)
    U = U3@U2@U1@U1@U2@U3
    U = sparse.dia_array(U)

    # Evolve phi over time.
    phis = [phi]
    norms = [np.linalg.norm(phi)]
    for i in range(n_iter):
        phi = U@phi
        if i%100 == 0:
            phis.append(phi)
            norms.append(np.linalg.norm(phi))

    # Animate
    def update_line(i, data, norms, line, ax):
        ax.title.set_text(f'frame = {i:03}, 1-norm={1 - norms[i]:.3e}')
        line.set_ydata(np.abs(data[i]))
        return line,

    # Scale V so we can put it in the same plot as phi.
    V_plot = V/np.max(V)*np.max(phis[0].real)*2

    fig, ax = plt.subplots()
    ax.title.set_text(f'frame = {0:03}, 1-norm={1 - norms[0]:.3e}')
    l, = ax.plot(x, np.abs(phis[0]), 'r-')
    ax.plot(x, V_plot)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$|\phi(x)|$')
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$V(x)$')
    line_ani = animation.FuncAnimation(fig, update_line, len(phis), fargs=(phis, norms, l, ax), interval=15,
                                       repeat=True)

    if save_ani:
        line_ani.save('potential_well.mp4')
    if show_ani:
        plt.show()


if __name__ == '__main__':
    """Uncomment functions to see the different examples.
    Saving the animation can take a rather long time."""

    parabolic_potential_well(show_ani=True, save_ani=False)
    # quantum_tunneling(show_ani=True, save_ani=False)
