import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import solve_continuous_are, solve_lyapunov

M, m, l, g, b, c = 10.0, 2.0, 2.0, 9.8, 0.1, 0.01
A = np.array([
    [0, 1, 0, 0],
    [0, -b/M, -m*g/M, c/(M*l)],
    [0, 0, 0, 1],
    [0, b/(M*l), (m+M)*g/(M*l), -(m+M)*c/(M*m*l**2)]
])

B_lqr = np.array([[0], [1/M], [0], [-1/(M*l)]])
max_x = 1.0
max_theta = np.deg2rad(20)
max_u = 100.0

Q = np.diag([1/max_x**2, 0.1, 1/max_theta**2, 0.1])
R = np.array([[1/max_u**2]])

P = solve_continuous_are(A, B_lqr, Q, R)
K = np.linalg.inv(R) @ B_lqr.T @ P

A_cl = A - B_lqr @ K
eigvals_cl = np.linalg.eigvals(A_cl)

print("\nВласні значення замкненої системи A_cl (A - BK):")
is_stable = True
for val in eigvals_cl:
    print(f"λ = {val.real:.4f} + {val.imag:.4f}j")
    if val.real >= 0:
        is_stable = False

if is_stable:
    print("Висновок: Система стійка (всі полюси в лівій півплощині)")
else:
    print("Висновок: Система нестійка")

plt.figure(figsize=(8, 6))
plt.scatter(eigvals_cl.real, eigvals_cl.imag, color='blue', marker='o', s=100, label='Полюси A_cl (LQR)')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.7)
limit = max(max(np.abs(eigvals_cl.real)), max(np.abs(eigvals_cl.imag)), 1) * 1.5
plt.xlim(-limit, limit/2) 
plt.ylim(-limit, limit)
plt.title('Розташування власних значень замкненої системи A_cl', fontsize=14)
plt.xlabel('Re', fontsize=12)
plt.ylabel('Im', fontsize=12)
plt.legend()
plt.show()

def get_derivatives(state, F):
    x, dx, theta, dtheta = state
    A_mat = np.array([
        [M + m, m * l * np.cos(theta)],
        [m * l * np.cos(theta), m * l ** 2]
    ])
    B_mat = np.array([
        F - b * dx + m * l * (dtheta ** 2) * np.sin(theta),
        -c * dtheta + m * g * l * np.sin(theta)
    ])
    ddx, ddtheta = np.linalg.solve(A_mat, B_mat)
    return np.array([dx, ddx, dtheta, ddtheta])

def run_simulation(initial_theta_deg, animate=True):
    dt = 0.03
    steps = 400
    S = np.array([0.0, 0.0, np.deg2rad(initial_theta_deg), 0.0])
    
    data = {"t": [], "x": [], "theta": [], "u": []}
    current_time = 0.0
    
    for _ in range(steps):
        data["t"].append(current_time)
        data["x"].append(S[0])
        data["theta"].append(S[2])
        
        u = -(K @ S)[0]
        u = np.clip(u, -max_u, max_u)
        data["u"].append(u)
    
        k1 = get_derivatives(S, u)
        k2 = get_derivatives(S + k1 * dt / 2, u)
        k3 = get_derivatives(S + k2 * dt / 2, u)
        k4 = get_derivatives(S + k3 * dt, u)
        S += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        current_time += dt

    if animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_aspect('equal')
        ax.set_ylim(-1.0, l + 1.0)
        ax.grid(True)
        ax.set_title(f"LQR Анімація: Початковий кут {initial_theta_deg}°")
        ax.plot([-100, 100], [-0.25, -0.25], color='black', lw=2)
        cart = patches.Rectangle((0, -0.25), 1.6, 0.5, fc='blue', ec='black')
        ax.add_patch(cart)
        pendulum, = ax.plot([], [], 'o-', lw=4, color='red', markersize=10)

        for i in range(0, steps, 2): 
            x_pos = data["x"][i]
            th = data["theta"][i]
            ax.set_xlim(x_pos - 5, x_pos + 5)
            cart.set_xy((x_pos - 0.8, -0.25))
            pendulum.set_data([x_pos, x_pos + l * np.sin(th)], [0.25, 0.25 + l * np.cos(th)])
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
        plt.ioff()
        plt.close(fig)
    
    return data

print("Симуляція 5 градусів...")
data5 = run_simulation(5, animate=True)

print("Симуляція 15 градусів...")
data15 = run_simulation(15, animate=True)

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('LQR: Порівняння динаміки (5° vs 15°)', fontsize=16)

axs[0].plot(data5["t"], data5["x"], label='5°', color='blue')
axs[0].plot(data15["t"], data15["x"], label='15°', color='red', linestyle='--')
axs[0].set_ylabel('Позиція візка x (м)')
axs[0].grid(True); axs[0].legend()

axs[1].plot(data5["t"], np.degrees(data5["theta"]), label='5°', color='blue')
axs[1].plot(data15["t"], np.degrees(data15["theta"]), label='15°', color='red', linestyle='--')
axs[1].set_ylabel('Кут theta (град)')
axs[1].grid(True); axs[1].legend()

axs[2].plot(data5["t"], data5["u"], label='5°', color='blue')
axs[2].plot(data15["t"], data15["u"], label='15°', color='red', linestyle='--')
axs[2].set_ylabel('Сила керування u (Н)')
axs[2].set_xlabel('Час (с)')
axs[2].grid(True); axs[2].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

Wc = solve_lyapunov(A_cl, -B_lqr @ B_lqr.T)
print("\n--- Аналіз Граміана керованості (діагональні елементи) ---")
states_names = ["Позиція (x)", "Швидкість (v)", "Кут (theta)", "Кутова швидкість (omega)"]
for name, val in zip(states_names, np.diag(Wc)):
    print(f"{name}: {val:.6e}")