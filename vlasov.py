import numpy as np

def lax_wendroff(f, courant):
    return f \
        - courant/2 * (np.roll(f, -1) - np.roll(f, 1)) \
        + courant**2/2 * (np.roll(f, -1) - 2*f + np.roll(f, 1))

def adv1d(*, system_length, velocity, init, ngrid, dt):
    f = init.copy()
    dx = system_length / ngrid
    c = velocity * dt / dx
    while True:
        yield f
        f = lax_wendroff(f, c)
