import numpy as np
from matplotlib import pyplot as plt, animation


if __name__ == '__main__':

    # Initial Values.
    _lambda = 1
    k = 1
    sigma = 6*_lambda
    delta = 0.1*_lambda
    tau = 0.05*_lambda
    alpha = (-1)/(4*pow(np.pi, 2)*pow(delta, 2))

    # Wave function.
    x0_val = 3*sigma
    grid_size = 1000
    x = np.arange(grid_size)*delta
    phi = np.exp(-2*np.pi*1j*k*x)*np.exp(((-1)*pow(x - x0_val, 2))/(2*pow(sigma, 2)))
    phi = phi/np.linalg.norm(phi)  # Normalize.
    lower_sup = x > x0_val - 3*sigma
    upper_sup = x < x0_val + 3*sigma
    sup = lower_sup & upper_sup
    phi[~sup] = 0

    # Matrix
    M = np.array([[np.cos(tau*abs(alpha)), (-1j)*(alpha/abs(alpha))*np.sin(tau*alpha)],
                  [(-1j)*np.sin(tau*abs(alpha)), np.cos(tau*abs(alpha))]])
    print(f'{M = !s}')

    phi_new = np.zeros_like(phi)
    n_iter = 5000
    phi_list = [phi]
    V = np.zeros_like(phi)  # Potential V.
    V[700:750] = 0.7
    for i in range(n_iter):
        # U1 * phi
        for j_even in range(0, grid_size, 2):
            phi_new[j_even: j_even + 2] = M@phi[j_even:j_even + 2]

        # U2 * phi
        for j_odd in range(1, grid_size - 1, 2):
            phi_new[j_odd: j_odd + 2] = M@phi_new[j_odd:j_odd + 2]

        u3 = np.exp(-1j*tau*alpha*(2 + 4*np.pi**2*delta**2*V))

        phi_new = u3*phi_new
        if i%100 == 0:
            phi_list.append(phi_new)
        phi = phi_new


    # PLot
    # plt.plot(x, phi_list[0], label = "First")
    # plt.plot(x, phi_list[-1], label = "Last")
    # plt.legend()
    # plt.show()

    # Animate
    def update_line(num, data, line, ax):
        ax.title.set_text(f'frame = {num}')
        line.set_ydata(data[num].real)
        return line,


    fig1, ax = plt.subplots()
    l, = plt.plot(x, phi_list[0].real, 'r-')
    plt.plot(x, V)
    line_ani = animation.FuncAnimation(fig1, update_line, len(phi_list), fargs=(phi_list, l, ax), interval=50, repeat=True)
    plt.show()
