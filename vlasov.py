import numpy as np
import filter

def lax_wendroff(f, courant):
    return f \
        - courant/2 * (np.roll(f, -1) - np.roll(f, 1)) \
        + courant**2/2 * (np.roll(f, -1) - 2*f + np.roll(f, 1))

def flux_limited_lax_wendroff(limiter):
    def lw(u, courant):
        diff = np.roll(u,-1) - u # diff[i] = u[i+1] -u[i]
        eps = 1e-100 # avoid zero division
        p = limiter(np.divide(np.roll(diff, 1), diff+eps)) #  p[i] = (u[i]-u[i-1])/(u[i+1]-u[i])
        F = u + p * (1-courant)/2 * diff
        return u + courant * (np.roll(F,1) - F)
    return lw

def adv1d(*, system_length, velocity, init, ngrid, dt):
    f = init.copy()
    dx = system_length / ngrid
    c = velocity * dt / dx
    advance = flux_limited_lax_wendroff(filter.superbee)
    while True:
        yield f
        f = advance(f, c)
