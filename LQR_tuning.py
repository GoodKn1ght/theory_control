# stabilizes up from 60 degrees with 1000 H maximum u
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import solve_continuous_are

M, m, l, g, b, c = 10.0, 2.0, 2.0, 9.8, 0.1, 0.01

def get_lqr_matrices_cramer(current_theta):
    th_deg = np.degrees(np.abs(current_theta))
    if th_deg < 15: 
        theta0 = 0.0
    elif th_deg < 37.5: 
        theta0 = np.deg2rad(30)
    else: 
        theta0 = np.deg2rad(45)
    s0, c0 = np.sin(theta0), np.cos(theta0)
    det = (M + m) * (m * l**2) - (m * l * c0)**2
    A = np.array([
        [0, 1, 0, 0],
        [0, -(b * m * l**2) / det, -(m**2 * l**2 * g * (c0**2 - s0**2)) / det, (c * m * l * c0) / det],
        [0, 0, 0, 1],
        [0, (m * l * c0 * b) / det, ((M + m) * m * g * l * c0) / det, -( (M + m) * c ) / det]
    ])
    B = np.array([
        [0],
        [(m * l**2) / det],
        [0],
        [(-m * l * c0) / det]
    ])
    
    return A, B

def get_derivatives(state, F):
    x, v, theta, omega = state
    M_mat = np.array([
        [M + m, m * l * np.cos(theta)],
        [m * l * np.cos(theta), m * l**2]
    ])
    F_vec = np.array([
        F - b * v + m * l * omega**2 * np.sin(theta),
        m * g * l * np.sin(theta) - c * omega
    ])
    accels = np.linalg.solve(M_mat, F_vec)
    return np.array([v, accels[0], omega, accels[1]])

def run_simulation(start_angle_deg):
    dt = 0.02
    steps = 600
    S = np.array([0.0, 0.0, np.deg2rad(start_angle_deg), 0.0])
    Q = np.diag([1.0, 0.1, 10.0, 0.1])
    R = np.array([[0.001]])
    
    history = {"t": [], "x": [], "theta": [], "u": []}
    
    print(f"Початок симуляції з {start_angle_deg} градусів...")
    
    for i in range(steps):
        A, B = get_lqr_matrices_cramer(S[2])
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        u = -(K @ S)[0]
        u = np.clip(u, -1000, 1000)
        
        history["t"].append(i * dt)
        history["x"].append(S[0])
        history["theta"].append(np.degrees(S[2]))
        history["u"].append(u)
        k1 = get_derivatives(S, u)
        k2 = get_derivatives(S + k1 * dt / 2, u)
        k3 = get_derivatives(S + k2 * dt / 2, u)
        k4 = get_derivatives(S + k3 * dt, u)
        S += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        if np.abs(S[2]) > np.pi:
            print("Маятник впав!")
            break
            
    return history

data = run_simulation(60)
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
axs[0].plot(data["t"], data["x"], label='Позиція візка')
axs[0].set_ylabel('x (м)'); axs[0].grid(True); axs[0].legend()

axs[1].plot(data["t"], data["theta"], label='Кут маятника', color='red')
axs[1].axhline(0, color='black', linestyle='--')
axs[1].set_ylabel('theta (град)'); axs[1].grid(True); axs[1].legend()

axs[2].plot(data["t"], data["u"], label='Сила керування u', color='green')
axs[2].set_ylabel('u (Н)'); axs[2].set_xlabel('Час (с)'); axs[2].grid(True); axs[2].legend()

plt.tight_layout()
plt.show()
def animate(data):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_aspect('equal')
    ax.set_ylim(-1, l + 1)
    cart = patches.Rectangle((0, -0.2), 1.0, 0.4, fc='blue', ec='black')
    ax.add_patch(cart)
    line, = ax.plot([], [], 'o-', lw=3, color='red', markersize=8)
    
    for i in range(0, len(data["t"]), 2):
        x = data["x"][i]
        th = np.radians(data["theta"][i])
        ax.set_xlim(x - 5, x + 5)
        cart.set_xy((x - 0.5, -0.2))
        line.set_data([x, x + l * np.sin(th)], [0, l * np.cos(th)])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    plt.ioff()
    plt.show()

animate(data)