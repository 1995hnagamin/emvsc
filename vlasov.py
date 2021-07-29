from dataclasses import dataclass, field
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


@dataclass
class Species:
    name: str
    q: float
    qm: float

@dataclass
class Vp2dConfig:
    species: list[Species]
    initial_distribution: np.ndarray
    background_charge_density: float
    system_length: float
    ngridx: float
    vmax: float
    ngridv: float
    dt: float


def divergence_matrix(n, dx):
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i - 1] = -1.0 / (2 * dx)
        D[i, (i + 1) % n] = 1.0 / (2 * dx)
    return D


def trapezoidal_rule(f, dx):
    return dx * (f.sum(axis=0) - (f[0, :] + f[-1, :]) / 2)


@njit("f8[:,:](f8[:,:], f8[:,:], f8, f8, f8[:], f8[:])", parallel=True)
def update(fs, fnew, dtdx, qmdtdv, v, E):
    ngridv, ngridx = fs.shape
    for iv in prange(ngridv):
        c = v[iv] * dtdx
        fnew[iv, :] = scheme.lax_wendroff_superbee_p(fs[iv, :], c)
    fs = fnew
    for ix in prange(ngridx):
        c = E[ix] * qmdtdv
        fnew[:, ix] = scheme.lax_wendroff_superbee_p(fs[:, ix], c)
    fs = fnew
    return fs


# 2-dimensional Vlasov-Poisson equation
def vp2d(config: Vp2dConfig):
    nspecies = len(config.species)
    f = config.initial_distribution.copy()
    dx = config.system_length / config.ngridx
    v = np.linspace(-config.vmax, config.vmax, config.ngridv, endpoint=False)
    dv = 2 * config.vmax / config.ngridv
    dt = config.dt
    eps0 = 1.0

    A = np.linalg.pinv(divergence_matrix(config.ngridx, dx))

    rho = np.empty(config.ngridx)
    E = np.empty(config.ngridx)
    fnew = np.empty((config.ngridv, config.ngridx))

    q = [species.q for species in config.species]
    qm = [species.qm for species in config.species]
    background_charge_density = config.background_charge_density

    while True:
        rho[:] = background_charge_density
        for s in range(nspecies):
            rho += q[s] * trapezoidal_rule(f[s], dv)
        E = A.dot(rho / eps0)
        yield (f, rho, E)

        for s in range(nspecies):
            f[s] = update(f[s], fnew, dt / dx, qm[s] * dt / dv, v, E)
