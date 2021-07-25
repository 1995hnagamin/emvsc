import numpy as np


# Lax-Wendroff scheme, 1D periodic boundary condition
def lax_wendroff_p(u, courant):
    p = np.empty(len(u) + 2)
    p[1:-1] = u
    p[0] = u[-1]
    p[-1] = u[0]
    return (
        u - courant / 2 * (p[2:] - p[:-2]) + courant ** 2 / 2 * (p[2:] - 2 * u + p[:-2])
    )


# Lax-Wendroff scheme with flux limiter, 1D periodic boundary condition
def flux_limited_lax_wendroff_p(limiter):
    eps = 1e-100  # avoid zero division

    def lw(u, courant):
        v = np.empty(len(u) + 4, dtype=object)
        v[2:-2] = u
        v[:2] = u[-2:]
        v[-2:] = u[:2]
        diff = v[1:] - v[:-1]
        if courant > 0:
            p = limiter(np.divide(diff[:-2], diff[1:-1] + eps))
            F = v[1:-2] + (1 - courant) / 2 * p * diff[1:-1]
            return v[2:-2] + courant * (F[:-1] - F[1:])
        else:
            p = limiter(np.divide(diff[2:], diff[1:-1] + eps))
            F = v[2:-1] - (1 + courant) / 2 * p * diff[1:-1]
            return v[2:-2] - courant * (F[1:] - F[:-1])

    return lw


# Lax-Wendroff scheme with superbee flux limiter, 1D periodic boundary condition
def lax_wendroff_superbee_p(u, courant):
    eps = 1e-100  # avoid zero division

    v = np.empty(len(u) + 4, dtype=object)
    v[2:-2] = u
    v[:2] = u[-2:]
    v[-2:] = u[:2]
    diff = v[1:] - v[:-1]
    if courant > 0:
        r = np.divide(diff[:-2], diff[1:-1] + eps)
        p = np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))
        F = v[1:-2] + (1 - courant) / 2 * p * diff[1:-1]
        return v[2:-2] + courant * (F[:-1] - F[1:])
    else:
        r = np.divide(diff[2:], diff[1:-1] + eps)
        p = np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))
        F = v[2:-1] - (1 + courant) / 2 * p * diff[1:-1]
        return v[2:-2] - courant * (F[1:] - F[:-1])
