import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import scipy.linalg
from scipy.linalg import expm
import matplotlib.animation as animation

G = 9.81
MASS = 1.0
DT = 0.05
T_END = 10.0

SENSOR_NOISE_POS = 0.15
SENSOR_NOISE_OMGA = 0.03
IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

def drone_dynamics(x, u):
    q = x[6:10]
    norm_q = np.linalg.norm(q)
    if norm_q < 1e-6: q = np.array([1.0, 0, 0, 0])
    else: q = q / norm_q
    
    omega = x[10:13]
    thrust_val, u_omega_dot = u[0], u[1:4]
    
    d_pos = x[3:6]
    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    accel = (1.0/MASS) * (rot.as_matrix() @ np.array([0, 0, thrust_val])) - np.array([0, 0, G])
    
    qw, qx, qy, qz = q
    wx, wy, wz = omega
    d_q = 0.5 * np.array([
        -qx*wx - qy*wy - qz*wz,
         qw*wx - qz*wy + qy*wz,
         qz*wx + qw*wy - qx*wz,
        -qy*wx + qx*wy + qw*wz
    ])
    d_omega = u_omega_dot
    return np.concatenate([d_pos, accel, d_q, d_omega])

def get_linearized_matrices(x, u):
    n, m = len(x), len(u)
    epsilon = 1e-6
    A, B = np.zeros((n, n)), np.zeros((n, m))
    f0 = drone_dynamics(x, u)
    for i in range(n):
        x_eps = x.copy(); x_eps[i] += epsilon
        A[:, i] = (drone_dynamics(x_eps, u) - f0) / epsilon
    for i in range(m):
        u_eps = u.copy(); u_eps[i] += epsilon
        B[:, i] = (drone_dynamics(x, u_eps) - f0) / epsilon
    return A, B

def discretize_system(A_cont, B_cont, dt):
    n, m = A_cont.shape[0], B_cont.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_cont
    M[:n, n:] = B_cont
    ExpM = expm(M * dt)
    return ExpM[:n, :n], ExpM[:n, n:]

def rk4_step(x, u, dt):
    k1 = drone_dynamics(x, u)
    k2 = drone_dynamics(x + dt/2 * k1, u)
    k3 = drone_dynamics(x + dt/2 * k2, u)
    k4 = drone_dynamics(x + dt * k3, u)
    x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    x_next[6:10] /= np.linalg.norm(x_next[6:10])
    return x_next
x_target = np.array([-4.0, 2.0, 8.0, 0,0,0, 1.0,0,0,0, 0,0,0], dtype=np.float64)
x_real = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
u_hover = np.array([MASS*G, 0.0, 0.0, 0.0], dtype=np.float64)
Q_lqr = np.diag([120, 120, 250, 20, 20, 20, 100, 100, 100, 10, 10, 10])
R_lqr = np.diag([0.5, 5.0, 5.0, 5.0]) 

x_hat = x_real[IDX].copy()
P_kf = np.eye(12) * 0.1
Q_kf = np.eye(12) * 1e-4

Cd = np.zeros((6, 12))
Cd[0:3, 0:3] = np.eye(3)    
Cd[3:6, 9:12] = np.eye(3)   
R_kf = np.diag([SENSOR_NOISE_POS**2]*3 + [SENSOR_NOISE_OMGA**2]*3)
A_f, B_f = get_linearized_matrices(x_real, u_hover)
Ad_init, Bd_init = discretize_system(A_f[np.ix_(IDX, IDX)], B_f[IDX, :], DT)
P_init = scipy.linalg.solve_discrete_are(Ad_init, Bd_init, Q_lqr, R_lqr)
K_lqr = np.linalg.inv(R_lqr + Bd_init.T @ P_init @ Bd_init) @ Bd_init.T @ P_init @ Ad_init

history = []
u = u_hover.copy()
dare_failures = 0

print(f"Політ (Pure SDRE). Ціль: {x_target[0:3]}...")

for t in np.arange(0, T_END, DT):
    history.append(x_real.copy())
    z_noisy = (Cd @ x_real[IDX]) + np.random.normal(0, [SENSOR_NOISE_POS]*3 + [SENSOR_NOISE_OMGA]*3)
    x_hat_full = np.zeros(13, dtype=np.float64)
    x_hat_full[IDX] = x_hat
    norm_sq = np.sum(x_hat[6:9]**2)
    x_hat_full[6] = np.sqrt(max(0.0, 1.0 - norm_sq))
    A_dyn, B_dyn = get_linearized_matrices(x_hat_full, u)
    Ad, Bd = discretize_system(A_dyn[np.ix_(IDX, IDX)], B_dyn[IDX, :], DT)
    try:
        P_dare = scipy.linalg.solve_discrete_are(Ad, Bd, Q_lqr, R_lqr)
        K_lqr = np.linalg.inv(R_lqr + Bd.T @ P_dare @ Bd) @ Bd.T @ P_dare @ Ad
    except np.linalg.LinAlgError:
        dare_failures += 1
    u_delta = u.copy(); u_delta[0] -= MASS*G
    x_pred = Ad @ x_hat + Bd @ u_delta
    P_pred = Ad @ P_kf @ Ad.T + Q_kf
    
    S = Cd @ P_pred @ Cd.T + R_kf
    K_gain = P_pred @ Cd.T @ np.linalg.inv(S)
    x_hat = x_pred + K_gain @ (z_noisy - Cd @ x_pred)
    P_kf = (np.eye(12) - K_gain @ Cd) @ P_pred
    err = x_hat - x_target[IDX]
    err[0:3] = np.clip(err[0:3], -0.6, 0.6) 
    err[3:6] = np.clip(err[3:6], -1.5, 1.5) 
    
    u_delta_new = -K_lqr @ err
    
    u = u_delta_new.copy()
    u[0] += MASS * G
    u = np.clip(u, [0, -5, -5, -5], [25, 5, 5, 5])

    x_real = rk4_step(x_real, u, DT)
    
    if int(t/DT) % 40 == 0:
        print(f"t={t:.1f}s | Pos: {x_real[0:3].round(2)} | DARE Fails: {dare_failures}")
history = np.array(history)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-6, 6]); ax.set_ylim([-6, 6]); ax.set_zlim([0, 10])
ax.scatter(*x_target[0:3], color='gold', s=150, marker='*', label='Target')

line, = ax.plot([], [], [], 'b', alpha=0.5, label='Flight Path')

ax_x = ax.quiver(0,0,0,0,0,0, color='r')
ax_y = ax.quiver(0,0,0,0,0,0, color='g')
ax_z = ax.quiver(0,0,0,0,0,0, color='b')

def update(num):
    global ax_x, ax_y, ax_z
    line.set_data(history[:num, 0], history[:num, 1])
    line.set_3d_properties(history[:num, 2])
    p, q = history[num, 0:3], history[num, 6:10]
    if np.linalg.norm(q) < 1e-6: q = np.array([1.0, 0, 0, 0])
    rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    
    ax_x.remove()
    ax_y.remove()
    ax_z.remove()
    ax_x = ax.quiver(p[0], p[1], p[2], rot[0,0], rot[1,0], rot[2,0], color='r', length=0.8)
    ax_y = ax.quiver(p[0], p[1], p[2], rot[0,1], rot[1,1], rot[2,1], color='g', length=0.8)
    ax_z = ax.quiver(p[0], p[1], p[2], rot[0,2], rot[1,2], rot[2,2], color='b', length=1.2)
    
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=DT*1000, blit=False)
plt.legend()
plt.title("Drone Flight Simulation: SDRE LQR")
plt.show()