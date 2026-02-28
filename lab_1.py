import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
M, m, l, g, b, c, F = 10.0, 2.0, 2.0, 9.8, 0.1, 0.01, 30.0
def get_derivatives(state, t):
    x, dx, theta, dtheta = state
    A = np.array([
        [M + m, m * l * np.cos(theta)],
        [m * l * np.cos(theta), m * l ** 2]
    ])
    B = np.array([
        F - b * dx + m * l * (dtheta ** 2) * np.sin(theta),
        -c * dtheta + m * g * l * np.sin(theta)
    ])
    ddx, ddtheta = np.linalg.solve(A, B)
    return np.array([dx, ddx, dtheta, ddtheta])
dt = 0.03
steps = 300
S = np.array([0.0, 10.0, 0.1, 0.0])
states = []

for _ in range(steps):
    states.append(S.copy())
    k1 = get_derivatives(S, 0)
    k2 = get_derivatives(S + k1 * dt / 2, 0)
    k3 = get_derivatives(S + k2 * dt / 2, 0)
    k4 = get_derivatives(S + k3 * dt, 0)
    S += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

states = np.array(states)
x_hist = states[:, 0]
theta_hist = states[:, 2]
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_aspect('equal')
ax.set_ylim(-1.5, l + 2.0)
ax.grid(True)
ax.set_title("Рух маятника на візку (Слідкуюча камера)")
ax.plot([-5000, 5000], [-0.25, -0.25], color='black', lw=2)
cart_width = 3.0
cart_height = 1.0
cart = patches.Rectangle((0, -0.25), cart_width, cart_height, fc='blue', ec='black')
ax.add_patch(cart)
pendulum, = ax.plot([], [], 'o-', lw=4, color='red', markersize=10)

for i in range(steps):
    x = x_hist[i]
    theta = theta_hist[i]
    pivot_y = -0.25 + cart_height
    ax.set_xlim(x - 7, x + 7)
    cart.set_xy((x - cart_width / 2, -0.25))
    pendulum.set_data([x, x + l * np.sin(theta)], [pivot_y, pivot_y + l * np.cos(theta)])
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)
plt.ioff()
plt.show()