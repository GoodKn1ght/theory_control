import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import scipy.linalg
import matplotlib.animation as animation

G = 9.81
MASS = 1.0
DT = 0.05
T_END = 10.0

def drone_dynamics(x, u):
    q = x[6:10]
    omega = x[10:13]
    thrust_val = u[0]
    u_omega_dot = u[1:4]

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

def rk4_step(x, u, dt):
    k1 = drone_dynamics(x, u)
    k2 = drone_dynamics(x + dt/2 * k1, u)
    k3 = drone_dynamics(x + dt/2 * k2, u)
    k4 = drone_dynamics(x + dt * k3, u)
    x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    x_next[6:10] /= np.linalg.norm(x_next[6:10])
    return x_next

x_target = np.array([
    -4.0, 2.0, 8.0, 
    0.0, 0.0, 0.0, 
    1, 0, 0, 0,
    0.0, 0.0, 0.0
])
Q = np.diag([100, 100, 200, 10, 10, 10, 80, 80, 80, 2, 2, 2])
R_mat = np.diag([0.05, 0.8, 0.8, 0.8])

x = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
u = np.array([MASS*G, 0.0, 0.0, 0.0])
history = []

print(f"Старт симуляції. Ціль: {x_target[0:3]}")
idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

for t in np.arange(0, T_END, DT):
    history.append(x.copy())
    A_full, B_full = get_linearized_matrices(x, u)
    A = A_full[np.ix_(idx, idx)]
    B = B_full[idx, :]
    Ad = np.eye(12) + A * DT
    Bd = B * DT

    try:
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R_mat)
        K = np.linalg.inv(R_mat + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
        err_full = x - x_target
        if np.dot(x[6:10], x_target[6:10]) < 0:
            err_full[6:10] = x[6:10] + x_target[6:10]
            
        err_reduced = err_full[idx]
        
        u_lqr = -K @ err_reduced
        u = u_lqr
        u[0] += MASS * G
    except np.linalg.LinAlgError:
        print(f"DARE failed at {t:.2f}")

    x = rk4_step(x, u, DT)
    
    if int(t/DT) % 20 == 0:
        print(f"t={t:.1f}s | Pos: {x[0:3].round(2)} | Quat_vec: {x[7:10].round(2)}")

history = np.array(history)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-5, 5]); ax.set_ylim([-5, 5]); ax.set_zlim([0, 10])
ax.scatter(*x_target[0:3], color='gold', s=100, marker='*', label='Target')

line, = ax.plot([], [], [], 'b', alpha=0.5)
ax_x = ax.quiver(0,0,0,0,0,0, color='r')
ax_y = ax.quiver(0,0,0,0,0,0, color='g')
ax_z = ax.quiver(0,0,0,0,0,0, color='b')

def update(num):
    global ax_x, ax_y, ax_z
    line.set_data(history[:num, 0], history[:num, 1])
    line.set_3d_properties(history[:num, 2])
    p, q = history[num, 0:3], history[num, 6:10]
    rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    ax_x.remove(); ax_y.remove(); ax_z.remove()
    ax_x = ax.quiver(p[0], p[1], p[2], rot[0,0], rot[1,0], rot[2,0], color='r', length=0.6)
    axis_y = ax.quiver(p[0], p[1], p[2], rot[0,1], rot[1,1], rot[2,1], color='g', length=0.6)
    axis_z = ax.quiver(p[0], p[1], p[2], rot[0,2], rot[1,2], rot[2,2], color='b', length=1.0)
    ax_y = axis_y; ax_z = axis_z
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=DT*1000, blit=False)
plt.show()