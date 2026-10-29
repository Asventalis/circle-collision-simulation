import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

# ============= Parameters =============
W, H = 8.0, 5.0
N = 8
r_min, r_max = 0.18, 0.32
dt = 0.005
STEPS_PER_FRAME = 5
FRAMES = 600
e = 0.95
mu_disk = 0.90   # disk–disk friction coefficient
mu_wall = 0.05   # disk–wall friction coefficient
speed_scale = 1.8
seed = 42
SPIN_MARK_RATIO = 0.80
# ======================================

rng = np.random.default_rng(seed)

r = rng.uniform(r_min, r_max, N)
m = np.pi * r**2
I = 0.5 * m * r**2
pos = np.zeros((N, 2))
vel = np.zeros((N, 2))
omega = np.zeros(N)
theta = np.zeros(N)

def tangent_from_normal(n):
    return np.array([-n[1], n[0]])

# ---------- Initialization ----------
def random_position(i, max_tries=6000):
    for _ in range(max_tries):
        x = rng.uniform(r[i], W - r[i])
        y = rng.uniform(r[i], H - r[i])
        cand = np.array([x, y])
        if i == 0:
            return cand
        d2 = np.sum((pos[:i] - cand)**2, axis=1)
        if np.all(d2 >= (r[:i] + r[i])**2):
            return cand
    raise RuntimeError("Cannot place disk without overlap")

for i in range(N):
    pos[i] = random_position(i)
    ang = rng.uniform(0, 2*np.pi)
    vmag = rng.uniform(0.5, 1.0) * speed_scale
    vel[i] = vmag * np.array([np.cos(ang), np.sin(ang)])
    omega[i] = rng.uniform(-5.0, 5.0)
    theta[i] = rng.uniform(0, 2*np.pi)

# ---------- Collision Solvers ----------
def resolve_disk_collisions(pos, vel, omega, r, m, I, e, mu):
    Nloc = len(r)
    for i in range(Nloc):
        for j in range(i+1, Nloc):
            d = pos[j] - pos[i]
            dist2 = float(d @ d)
            min_d = r[i] + r[j]
            if dist2 < min_d**2:
                dist = np.sqrt(max(dist2, 1e-24))
                n = d / dist
                t = tangent_from_normal(n)

                overlap = (min_d - dist)
                inv_sum = 1.0/m[i] + 1.0/m[j]
                pos[i] -= n * (overlap * (1.0/m[i]) / inv_sum)
                pos[j] += n * (overlap * (1.0/m[j]) / inv_sum)

                vij = vel[j] - vel[i]
                v_n = vij @ n
                v_t = vij @ t + (-omega[j]*r[j] - omega[i]*r[i])

                if v_n >= 0:
                    continue

                jn = -(1.0 + e) * v_n / (1.0/m[i] + 1.0/m[j])

                inv_meff_t = (1.0/m[i] + 1.0/m[j]) + (r[i]**2)/I[i] + (r[j]**2)/I[j]
                jt_star = - v_t / inv_meff_t
                jt_max = mu * abs(jn)
                jt = jt_star if abs(jt_star) <= jt_max else -jt_max * np.sign(v_t)

                vel[i] -= (jn/m[i]) * n + (jt/m[i]) * t
                vel[j] += (jn/m[j]) * n + (jt/m[j]) * t

                omega[i] -= (jt * r[i]) / I[i]
                omega[j] -= (jt * r[j]) / I[j]

def apply_wall_contact(i, n, mu):
    t = tangent_from_normal(n)
    v = vel[i]
    v_n = v @ n
    if v_n >= 0:
        return

    jn = -(1.0 + e) * m[i] * v_n

    v_t = v @ t - omega[i] * r[i]
    meff_t = m[i] / 3.0
    jt_star = - meff_t * v_t
    jt_max = mu * abs(jn)
    jt = jt_star if abs(jt_star) <= jt_max else -jt_max * np.sign(v_t)

    vel[i] = v + (jn/m[i]) * n + (jt/m[i]) * t
    omega[i] -= (jt * r[i]) / I[i]

def resolve_wall_collisions(pos, vel, omega, r, m, I, W, H, e, mu):
    for i in range(len(r)):
        if pos[i,0] - r[i] < 0:
            pos[i,0] = r[i]
            apply_wall_contact(i, np.array([1.0, 0.0]), mu)
        if pos[i,0] + r[i] > W:
            pos[i,0] = W - r[i]
            apply_wall_contact(i, np.array([-1.0, 0.0]), mu)
        if pos[i,1] - r[i] < 0:
            pos[i,1] = r[i]
            apply_wall_contact(i, np.array([0.0, 1.0]), mu)
        if pos[i,1] + r[i] > H:
            pos[i,1] = H - r[i]
            apply_wall_contact(i, np.array([0.0, -1.0]), mu)

# ---------- Physics step ----------
TWO_PI = 2*np.pi
def step():
    global pos, vel, omega, theta
    pos += vel * dt
    theta = (theta + omega * dt) % TWO_PI
    resolve_wall_collisions(pos, vel, omega, r, m, I, W, H, e, mu_wall)
    resolve_disk_collisions(pos, vel, omega, r, m, I, e, mu_disk)

# ---------- Animation ----------
fig, ax = plt.subplots(figsize=(W/1.3, H/1.3), dpi=100)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])

circles = [Circle(pos[i], r[i], alpha=0.95, facecolor="#a8d5ff", edgecolor="k", lw=0.8)
           for i in range(N)]
for c in circles: ax.add_patch(c)

lines = [ax.plot([], [], color="k", lw=1.6, solid_capstyle="round", zorder=3)[0]
         for _ in range(N)]

def init():
    for i, c in enumerate(circles):
        c.center = pos[i]
        lines[i].set_data([], [])
    return circles + lines

def animate(frame):
    for _ in range(STEPS_PER_FRAME):
        step()
    for i, c in enumerate(circles):
        c.center = pos[i]
        ang = theta[i]
        x0, y0 = pos[i]
        L = SPIN_MARK_RATIO * r[i]
        x1 = x0 + L * np.cos(ang)
        y1 = y0 + L * np.sin(ang)
        lines[i].set_data([x0, x1], [y0, y1])
    return circles + lines

interval_ms = 1000.0 * dt * STEPS_PER_FRAME
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=FRAMES, interval=interval_ms, blit=True)
plt.show()
