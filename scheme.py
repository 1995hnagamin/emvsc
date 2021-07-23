import numpy as np


# Lax-Wendroff scheme, 1D periodic boundary condition
def lax_wendroff_p(u, courant):
    return (
        u
        - courant / 2 * (np.roll(u, -1) - np.roll(u, 1))
        + courant ** 2 / 2 * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
    )


# Lax-Wendroff scheme with flux limiter, 1D periodic boundary condition
def flux_limited_lax_wendroff_p(limiter):
    eps = 1e-100  # avoid zero division

    def lw(u, courant):
        diff = np.roll(u, -1) - u  # diff[i] = u[i+1] -u[i]
        p = limiter(
            np.divide(np.roll(diff, 1), diff + eps)
        )  #  p[i] = (u[i]-u[i-1])/(u[i+1]-u[i])
        F = u + p * (1 - courant) / 2 * diff
        return u + courant * (np.roll(F, 1) - F)

    return lw
