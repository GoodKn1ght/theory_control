import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation

INITIAL_THRUST = 10.5  
CONTROL_WX = 0.1      
CONTROL_WY = 0.1      
CONTROL_WZ = 0.1     

G = 9.81
MASS = 1.0
DT = 0.05      
T_END = 5.0

def drone_dynamics(x, u):
    q = x[6:10]      
    omega = x[10:13] 
    thrust_val = u[0]
    
    d_pos = x[3:6]
    
    q_scipy = [q[1], q[2], q[3], q[0]] 
    rot = R.from_quat(q_scipy)
    r_matrix = rot.as_matrix()
    
    accel = (1.0/MASS) * (r_matrix @ np.array([0, 0, thrust_val])) - np.array([0, 0, G])
    
    qw, qx, qy, qz = q
    wx, wy, wz = omega
    d_q = 0.5 * np.array([
        -qx*wx - qy*wy - qz*wz,
         qw*wx - qz*wy + qy*wz,
         qz*wx + qw*wy - qx*wz,
        -qy*wx + qx*wy + qw*wz
    ])
    
    return np.concatenate([d_pos, accel, d_q, np.zeros(3)])

def rk4_step(x, u, dt):
    k1 = drone_dynamics(x, u)
    k2 = drone_dynamics(x + dt/2 * k1, u)
    k3 = drone_dynamics(x + dt/2 * k2, u)
    k4 = drone_dynamics(x + dt * k3, u)
    x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    x_next[6:10] /= np.linalg.norm(x_next[6:10]) 
    return x_next

x = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, CONTROL_WX, CONTROL_WY, CONTROL_WZ])
u = np.array([INITIAL_THRUST, CONTROL_WX, CONTROL_WY, CONTROL_WZ])

history = []
for _ in np.arange(0, T_END, DT):
    history.append(x.copy())
    x = rk4_step(x, u, DT)
history = np.array(history)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([0, 10])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

line, = ax.plot([], [], [], 'b', alpha=0.5)
axis_x = ax.quiver(0, 0, 0, 0, 0, 0, color='r')
axis_y = ax.quiver(0, 0, 0, 0, 0, 0, color='g')
axis_z = ax.quiver(0, 0, 0, 0, 0, 0, color='b')

def update(num):
    line.set_data(history[:num, 0], history[:num, 1])
    line.set_3d_properties(history[:num, 2])
    
    pos = history[num, 0:3]
    q = history[num, 6:10]
    rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    
    global axis_x, axis_y, axis_z
    axis_x.remove()
    axis_y.remove()
    axis_z.remove()
    
    axis_x = ax.quiver(pos[0], pos[1], pos[2], rot[0,0], rot[1,0], rot[2,0], color='r', length=0.6)
    axis_y = ax.quiver(pos[0], pos[1], pos[2], rot[0,1], rot[1,1], rot[2,1], color='g', length=0.6)
    axis_z = ax.quiver(pos[0], pos[1], pos[2], rot[0,2], rot[1,2], rot[2,2], color='b', length=1.0)
    
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=DT*1000, blit=False)
plt.show()