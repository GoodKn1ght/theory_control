import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# при опорі 0.1 та амортизації 0.01 гарно працює P = 150 I = 0 D = 40 window size = байдуже
# при опорі 100 та амортизації 10 гарно працює P = 600 I = 1000 D = 0 window size = steps
# при опорі 0.1 та амортизації 0.01 гарно працює P = 150 I = 50 D = 25 window size = 20
M, m, l, g, b, c = 10.0, 2.0, 2.0, 9.8, 0.1, 0.01

K_p = 150.0
K_i = 50.0
K_d = 25.0

def get_derivatives(state, F):
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
steps = 400
S = np.array([0.0, 10.0, -0.2, 0.0])
states = []

F_hist = []
time_hist = []
current_time = 0.0
error_history = []
window_size = 20

for _ in range(steps):
    states.append(S.copy())
    x, dx, theta, dtheta = S
    error = theta
    error_history.append(error * dt)
    if len(error_history) > window_size:
        error_history.pop(0)
    integral_part = sum(error_history)
    F = K_p * error + K_i * integral_part + K_d * dtheta

    F_hist.append(F)
    time_hist.append(current_time)

    k1 = get_derivatives(S, F)
    k2 = get_derivatives(S + k1 * dt / 2, F)
    k3 = get_derivatives(S + k2 * dt / 2, F)
    k4 = get_derivatives(S + k3 * dt, F)
    S += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    current_time += dt

states = np.array(states)
time_hist = np.array(time_hist)
F_hist = np.array(F_hist)

x_hist = states[:, 0]
theta_hist = states[:, 2]
dtheta_hist = states[:, 3]

plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_aspect('equal')
ax.set_ylim(-1.5, l + 1.5)
ax.grid(True)
ax.set_title("Обернений маятник з ПІД-регулятором")

ax.plot([-500, 500], [-0.25, -0.25], color='black', lw=2)

cart = patches.Rectangle((0, -0.25), 2.0, 0.5, fc='blue', ec='black')
ax.add_patch(cart)
pendulum, = ax.plot([], [], 'o-', lw=4, color='red', markersize=10)

for i in range(steps):
    x = x_hist[i]
    theta = theta_hist[i]

    pivot_y = 0.25

    ax.set_xlim(x - 5, x + 5)
    cart.set_xy((x - 1.0, -0.25))
    pendulum.set_data([x, x + l * np.sin(theta)], [pivot_y, pivot_y + l * np.cos(theta)])

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

plt.ioff()
plt.close(fig)
print("Побудова графіків динаміки...")
fig2, axs = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Аналіз динаміки оберненого маятника', fontsize=16)
axs[0, 0].plot(time_hist, theta_hist, 'r-', lw=2)
axs[0, 0].set_title('Кут маятника $\\theta(t)$')
axs[0, 0].set_xlabel('Час (с)')
axs[0, 0].set_ylabel('Кут (рад)')
axs[0, 0].grid(True)
axs[0, 0].axhline(0, color='black', linestyle='--', lw=1)
axs[0, 1].plot(time_hist, dtheta_hist, 'm-', lw=2)
axs[0, 1].set_title('Кутова швидкість $\\dot{\\theta}(t)$')
axs[0, 1].set_xlabel('Час (с)')
axs[0, 1].set_ylabel('Швидкість (рад/с)')
axs[0, 1].grid(True)
axs[0, 1].axhline(0, color='black', linestyle='--', lw=1)
axs[1, 0].plot(time_hist, x_hist, 'b-', lw=2)
axs[1, 0].set_title('Положення візка $x(t)$')
axs[1, 0].set_xlabel('Час (с)')
axs[1, 0].set_ylabel('Позиція (м)')
axs[1, 0].grid(True)
axs[1, 1].plot(time_hist, F_hist, 'g-', lw=2)
axs[1, 1].set_title('Прикладена сила керування $F(t)$')
axs[1, 1].set_xlabel('Час (с)')
axs[1, 1].set_ylabel('Сила (Н)')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()