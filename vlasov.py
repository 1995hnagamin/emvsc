import numpy as np
import limiter

def lax_wendroff(u, courant):
    return u \
        - courant/2 * (np.roll(u, -1) - np.roll(u, 1)) \
        + courant**2/2 * (np.roll(u, -1) - 2*u + np.roll(u, 1))

def flux_limited_lax_wendroff(limiter):
    eps = 1e-100 # avoid zero division
    def lw(u, courant):
        diff = np.roll(u,-1) - u # diff[i] = u[i+1] -u[i]
        p = limiter(np.divide(np.roll(diff, 1), diff+eps)) #  p[i] = (u[i]-u[i-1])/(u[i+1]-u[i])
        F = u + p * (1-courant)/2 * diff
        return u + courant * (np.roll(F,1) - F)
    return lw

def adv1d(*, system_length, velocity, init, ngrid, dt):
    u = init.copy()
    dx = system_length / ngrid
    c = velocity * dt / dx
    advance = flux_limited_lax_wendroff(limiter.superbee)
    while True:
        yield u
        u = advance(u, c)

# 2-dimensional Vlasov-Poisson equation
def vp2d(*, q, m, ion_density, system_length, vmax, init, ngridx, ngridv, dt):
    f = init
    dx = system_length / ngridx
    v = np.linspace(-vmax, vmax, ngridv, endpoint=False)
    dv = 2 * vmax / ngridv
    advance = flux_limited_lax_wendroff(limiter.superbee)
    eps0 = 1.

    A = np.zeros((ngridx, ngridx))
    for i in range(ngridx):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, (i+1)%ngridx] = 1

    rho = np.empty(ngridx)
    phi = np.empty(ngridx)
    E = np.empty(ngridx)

    while True:
        rho = q * (ion_density - f.sum(axis=0))
        phi = np.linalg.solve(A, -rho/eps0)
        E = -(np.roll(phi,-1) - np.roll(phi,1)) / (2*dx)
        yield f

        fnew = np.zeros_like(f)
        for ix in range(ngridx):
            c = q * E[ix] / m * dt / dv
            fnew[ix,:] = advance(f[ix,:], c)
        f = fnew
        for iv in range(ngridv):
            c = v[iv] * dt / dx
            fnew[:,iv] = advance(f[:,iv], c)
        f = fnew
