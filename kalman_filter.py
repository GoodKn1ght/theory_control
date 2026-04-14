import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

# --- ПАРАМЕТРИ ---
M, m, l, g, b, c = 10.0, 2.0, 2.0, 9.8, 0.1, 0.01
dt = 0.01
start_angle_deg = 25

A = np.array([
    [0, 1, 0, 0],
    [0, -b/M, -m*g/M, c/(M*l)],
    [0, 0, 0, 1],
    [0, b/(M*l), (m+M)*g/(M*l), -(m+M)*c/(M*m*l**2)]
])
B_lqr = np.array([[0], [1/M], [0], [-1/(M*l)]])
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
D = np.zeros((2, 1))

Ad, Bd, Cd, _, _ = cont2discrete((A, B_lqr, C, D), dt)

Q_lqr = np.diag([10.0, 1.0, 10.0, 1.0])
R_lqr = np.array([[0.01]])
P_dare = solve_discrete_are(Ad, Bd, Q_lqr, R_lqr)
K_lqr = np.linalg.inv(R_lqr + Bd.T @ P_dare @ Bd) @ Bd.T @ P_dare @ Ad

eigs = np.abs(np.linalg.eigvals(Ad - Bd @ K_lqr))
print(f"Власні значення: {eigs.round(4)}")
assert np.all(eigs < 1.0), "LQR не стабілізує систему!"

# --- КАЛМАН ---
SENSOR_NOISE_X     = 0.5
SENSOR_NOISE_THETA = 0.05

R_kf = np.diag([SENSOR_NOISE_X**2, SENSOR_NOISE_THETA * 5])
Q_kf = np.diag([1e-4, 1e-4, 1e-4, 1e-4])

def kalman_step(y_noisy, u_prev, x_hat_old, P_old):
    x_pred = Ad @ x_hat_old + Bd * u_prev
    P_pred = Ad @ P_old @ Ad.T + Q_kf
    S = Cd @ P_pred @ Cd.T + R_kf
    K_gain = P_pred @ Cd.T @ np.linalg.inv(S)
    x_hat_new = x_pred + K_gain @ (y_noisy - Cd @ x_pred)
    P_new = (np.eye(4) - K_gain @ Cd) @ P_pred
    return x_hat_new, P_new

def get_derivatives(state, F):
    x, dx, theta, dtheta = state.flatten()
    A_mat = np.array([
        [M + m, m * l * np.cos(theta)],
        [m * l * np.cos(theta), m * l**2]
    ])
    B_mat = np.array([
        F - b * dx + m * l * dtheta**2 * np.sin(theta),
        -c * dtheta + m * g * l * np.sin(theta)
    ])
    accels = np.linalg.solve(A_mat, B_mat)
    return np.array([[dx], [accels[0]], [dtheta], [accels[1]]])

# --- СИМУЛЯЦІЯ ---
def run_simulation(initial_deg):
    steps = 1000
    S_real = np.array([[0.0], [0.0], [np.deg2rad(initial_deg)], [0.0]])
    x_hat = S_real.copy()
    P_kf = np.eye(4) * 0.1
    u = 0.0

    history = {
        "t": [], 
        "x_real": [], "x_est": [], "x_noisy": [],
        "th_real": [], "th_est": [], "th_noisy": []
    }

    for i in range(steps):
        noise = np.array([
            [np.random.normal(0, SENSOR_NOISE_X)],
            [np.random.normal(0, SENSOR_NOISE_THETA)]
        ])
        y_noisy = Cd @ S_real + noise

        x_hat, P_kf = kalman_step(y_noisy, u, x_hat, P_kf)
        u = float(np.clip(-(K_lqr @ x_hat)[0, 0], -1000, 1000))

        history["t"].append(i * dt)
        history["x_real"].append(S_real[0, 0])
        history["x_est"].append(x_hat[0, 0])
        history["x_noisy"].append(y_noisy[0, 0])
        history["th_real"].append(np.degrees(S_real[2, 0]))
        history["th_est"].append(np.degrees(x_hat[2, 0]))
        history["th_noisy"].append(np.degrees(y_noisy[1, 0]))

        k1 = get_derivatives(S_real, u)
        k2 = get_derivatives(S_real + k1 * dt / 2, u)
        k3 = get_derivatives(S_real + k2 * dt / 2, u)
        k4 = get_derivatives(S_real + k3 * dt, u)
        S_real += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return history

# --- АНІМАЦІЯ ---
def animate_results(data):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_aspect('equal')

    cart = patches.Rectangle((0, -0.2), 1.0, 0.4, fc='steelblue', ec='black', lw=1.5)
    ax.add_patch(cart)
    line_real, = ax.plot([], [], 'o-', lw=3, color='limegreen',
                         markersize=8, label='Реальний')
    line_est,  = ax.plot([], [], '--', lw=2, color='cyan',
                         label='Оцінка Калмана')
    ax.legend(loc='upper right')

    for i in range(0, len(data["t"]), 5):
        x  = data["x_real"][i]
        th = np.radians(data["th_real"][i])
        xe = data["x_est"][i]
        te = np.radians(data["th_est"][i])

        ax.set_xlim(x - 5, x + 5)
        ax.set_ylim(-0.5, 2.5)
        cart.set_xy((x - 0.5, -0.2))
        line_real.set_data([x,  x  + l * np.sin(th)], [0, l * np.cos(th)])
        line_est.set_data( [xe, xe + l * np.sin(te)], [0, l * np.cos(te)])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff()
    plt.close(fig)

# --- ГРАФІКИ ---
def plot_results(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Kalman Filter + Discrete LQR", fontsize=14, fontweight='bold')
    t = data["t"]

    # Позиція
    ax1.plot(t, data["x_noisy"], color='tomato',    alpha=0.35, lw=0.8,
             label='Датчик (зашумлений)')
    ax1.plot(t, data["x_real"], color='limegreen',  lw=2,
             label='Істина')
    ax1.plot(t, data["x_est"],  color='dodgerblue', lw=2, linestyle='--',
             label='Калман')
    ax1.set_ylabel("Позиція x, м")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Кут
    ax2.plot(t, data["th_noisy"], color='tomato',    alpha=0.35, lw=0.8,
             label='Датчик (зашумлений)')
    ax2.plot(t, data["th_real"], color='limegreen',  lw=2,
             label='Істина')
    ax2.plot(t, data["th_est"],  color='dodgerblue', lw=2, linestyle='--',
             label='Калман')
    ax2.set_ylabel("Кут θ, градуси")
    ax2.set_xlabel("Час, с")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- ЗАПУСК ---
data = run_simulation(start_angle_deg)
animate_results(data)
plot_results(data)