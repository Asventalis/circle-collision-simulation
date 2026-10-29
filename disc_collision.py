import numpy as np
import matplotlib.pyplot as plt

def disc_collision(m1, R1, v1, w1,
                   m2, R2, v2, w2,
                   e=0.9, mu_s=0.5, mu_k=0.3,
                   x1=np.array([0.,0.]), x2=np.array([1.,0.])):
    I1 = 0.5 * m1 * R1**2
    I2 = 0.5 * m2 * R2**2
    
    n = x2 - x1
    n = n / np.linalg.norm(n)
    t = np.array([-n[1], n[0]])
    r1c = R1 * n
    r2c = -R2 * n
    ez = np.array([0.,0.,1.])
    
    u1 = v1 + np.cross(w1*ez, np.append(r1c,0))[:2]
    u2 = v2 + np.cross(w2*ez, np.append(r2c,0))[:2]
    g = u2 - u1
    g_n = np.dot(g, n)
    g_t = np.dot(g, t)
    
    if g_n >= 0:
        return v1, w1, v2, w2
    
    Jn = -(1+e)*g_n / (1/m1 + 1/m2)
    Kt = 1/m1 + 1/m2 + (R1**2)/I1 + (R2**2)/I2
    Jt_stick = - g_t / Kt
    
    if abs(Jt_stick) <= mu_s * Jn:
        Jt = Jt_stick
    else:
        Jt = - mu_k * Jn * np.sign(g_t)
    
    J = Jn * n + Jt * t
    
    v1p = v1 - J/m1
    v2p = v2 + J/m2
    w1p = w1 - np.cross(np.append(r1c,0), np.append(J,0))[2]/I1
    w2p = w2 + np.cross(np.append(r2c,0), np.append(J,0))[2]/I2
    
    return v1p, w1p, v2p, w2p

# --- ESEMPIO GRAFICO ---
m1, R1 = 1.0, 0.5
m2, R2 = np.inf, 0
v1 = np.array([1.0, 0.3])
w1 = 5.0
v2 = np.array([-0.5, -0.2])*0
w2 = 0
x1 = np.array([0.,0.])
x2 = np.array([1.4,1.0])

v1p, w1p, v2p, w2p = disc_collision(m1,R1,v1,w1,m2,R2,v2,w2,
                                    e=1, mu_s=0, mu_k=0,
                                    x1=x1, x2=x2)

# --- Plot ---
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect("equal")
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 3)

# Disegna i dischi
circle1 = plt.Circle(x1, R1, fill=False, color="blue", lw=2, label="Disco 1")
circle2 = plt.Circle(x2, R2, fill=False, color="red", lw=2, label="Disco 2")
ax.add_patch(circle1)
ax.add_patch(circle2)

# Vettori velocità prima
ax.quiver(*x1, *v1, angles='xy', scale_units='xy', scale=1, color="blue", alpha=0.5)
ax.quiver(*x2, *v2, angles='xy', scale_units='xy', scale=1, color="red", alpha=0.5)

# Vettori velocità dopo
ax.quiver(*x1, *v1p, angles='xy', scale_units='xy', scale=1, color="blue", lw=2)
ax.quiver(*x2, *v2p, angles='xy', scale_units='xy', scale=1, color="red", lw=2)

# Testo con velocità angolari
ax.text(x1[0], x1[1]-1.0, f"w1-={w1:.2f}\nw1+={w1p:.2f}", color="blue")
ax.text(x2[0], x2[1]-1.0, f"w2-={w2:.2f}\nw2+={w2p:.2f}", color="red")

ax.legend()
ax.set_title("Collisione di due dischi con attrito (pre e post urto)")
plt.show()
