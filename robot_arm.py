import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Robot Arm Parameters ---
L1 = 1.0  # Length of link 1
L2 = 0.8  # Length of link 2

# --- Inverse Kinematics Function ---
def inverse_kinematics(x, y):
    d = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    d = np.clip(d, -1, 1)
    theta2 = np.arctan2(-np.sqrt(1 - d**2), d)
    theta1 = np.arctan2(y, x) - np.arctan2(
        L2 * np.sin(theta2),
        L1 + L2 * np.cos(theta2)
    )
    return theta1, theta2

# --- Forward Kinematics Function ---
def forward_kinematics(theta1, theta2):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return (0, x1, x2), (0, y1, y2)

# --- Target Points for arm to reach ---
targets = [
    (1.2, 0.5),
    (0.5, 1.2),
    (-0.8, 0.8),
    (-1.2, 0.3),
    (0.0, 1.3),
    (1.0, -0.8),
    (1.5, 0.0)
]

# --- Setup Plot ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('2D Robot Arm - Inverse Kinematics Simulation')

line, = ax.plot([], [], 'o-', lw=4, color='blue', markersize=10)
target_dot, = ax.plot([], [], 'r*', markersize=15)
trace, = ax.plot([], [], 'g--', lw=1, alpha=0.5)

trace_x, trace_y = [], []

def animate(i):
    target = targets[i % len(targets)]
    tx, ty = target

    theta1, theta2 = inverse_kinematics(tx, ty)
    xs, ys = forward_kinematics(theta1, theta2)

    line.set_data(xs, ys)
    target_dot.set_data([tx], [ty])

    trace_x.append(xs[2])
    trace_y.append(ys[2])
    trace.set_data(trace_x, trace_y)

    return line, target_dot, trace

ani = animation.FuncAnimation(
    fig, animate, frames=len(targets),
    interval=800, blit=True, repeat=True
)

plt.tight_layout()
plt.show()