"""
an updated version of Levenberg-Marquardt method,
incorporating with some
"""
import numpy as np


def eval_jac(fun, p, diff_p):
    return


def check_bounds(p, bounds):
    """ check if p is in bounds """
    return


def lm(fun, p0, diff_p, x_mask,
       xtol, ftol, gtol,
       bounds, kmax=1000, verbose=True, args=(), tau=1e-3):

    ndim = len(p0)

    # initialization
    p = np.copy(p0) # copy initial parameters
    k = 0  # number of iteration
    nu = 2

    # evaluate Jacobian
    J = eval_jac(fun, p, diff_p)

    A = J.T @ J  # approximation of Hessian matrix
    eps = fun(p0, *args)
    g = J.T @ eps

    stop = norm(g) <= gtol
    mu = tau * np.max(np.diag(A))
    while not stop and k<kmax:
        k += 1
        # solve step
        dp = np.linalg.solve(A+np.identity(ndim)*mu, g)
        if norm(dp) <= xtol:
            # xtol satisfied
            break
        else:
            p_new = p+dp
            rho = eps

        if np.norm(dx) < xtol:
            break
        else:
            pass

    return
