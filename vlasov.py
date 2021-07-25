import numpy as np
from numba import njit, prange
import limiter
import scheme


def adv1d(*, system_length, velocity, init, ngrid, dt):
    u = init.copy()
    dx = system_length / ngrid
    c = velocity * dt / dx
    advance = scheme.flux_limited_lax_wendroff_p(limiter.superbee)
    while True:
        yield u
        u = advance(u, c)


def divergence_matrix(n, dx):
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i - 1] = -1.0 / (2 * dx)
        D[i, (i + 1) % n] = 1.0 / (2 * dx)
    return D


def trapezoidal_rule(f, dx):
    return dx * (f.sum(axis=0) - (f[0, :] + f[-1, :]) / 2)


# 2-dimensional Vlasov-Poisson equation
def vp2d(*, q, qm, ion_density, system_length, vmax, init, ngridx, ngridv, dt):
    nspecies = len(q)
    f = init.copy()
    dx = system_length / ngridx
    v = np.linspace(-vmax, vmax, ngridv, endpoint=False)
    dv = 2 * vmax / ngridv
    advance = scheme.lax_wendroff_superbee_p
    eps0 = 1.0

    A = np.linalg.pinv(divergence_matrix(ngridx, dx))

    rho = np.empty(ngridx)
    E = np.empty(ngridx)
    fnew = np.empty((ngridv, ngridx))

    @njit(parallel=True)
    def update(fs, fnew, qm, advance):
        for iv in prange(ngridv):
            c = v[iv] * dt / dx
            fnew[iv, :] = advance(fs[iv, :], c)
        fs = fnew
        for ix in prange(ngridx):
            c = qm * E[ix] * dt / dv
            fnew[:, ix] = advance(fs[:, ix], c)
        fs = fnew
        return fs

    while True:
        rho.fill(ion_density)
        for s in range(nspecies):
            rho += q[s] * trapezoidal_rule(f[s], dv)
        E = A.dot(rho / eps0)
        yield (f, rho, E)

        for s in range(nspecies):
            f[s] = update(f[s], fnew, qm[s], advance)
