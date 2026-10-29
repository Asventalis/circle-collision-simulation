# Rigid Disk Collisions

A 2D physics toy where spinning disks of random radius bounce inside a box. Collisions are resolved with an impulse-based model: the normal impulse uses a coefficient of restitution e, and the tangential impulse is limited by Coulomb friction (stick if |Jt| <= mu |Jn|, otherwise slide), so angular velocity is exchanged with linear velocity on contact. Mass and moment of inertia (I = m r^2 / 2) follow from the radius; a short radial mark on each disk shows its rotation.

## How to run
Python 3 with `numpy` and `matplotlib`. `python collision.py` opens the live animation (8 disks, parameters at the top of the file). `python disc_collision.py` computes and plots a single pre/post collision between two disks (or a disk and a fixed wall, using infinite mass).

## Notes
- Integration is explicit Euler with positional overlap correction, so very fast disks can tunnel through each other.
- `disc_collision.py` distinguishes static and kinetic friction coefficients; `collision.py` uses a single coefficient per contact type.
