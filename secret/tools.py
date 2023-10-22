from .function_tools import *
from .spectral_solve import *


def get_uT(K=129):
    return load_end_state(K=K)


def get_a():
    w = angular_velocity_field()
    ax = lambda x, y: w(x, y) * (y-0.5)   # 3 is nice setting for the test
    ay = lambda x, y: -w(x, y) * (x-0.5)
    return ax, ay


def solve_slow(u0, ax, ay, nu, T, N=100, callback=None):
    u0 = u0.eval_grid()
    uT = solve_discrete(u0, lambda x,y: -ax(x, y), lambda x,y: -ay(x,y), lam=nu, N=N, T=T, callback=callback)
    return to_basis(uT)


def solve_fast(u0, upsample_factor=2):
    u0 = u0.change_dim(upsample_factor*u0.xDim, upsample_factor*u0.yDim)
    uT = get_sol_analytic(u0, angular_velocity_field(), 1.)
    return uT.change_dim(u0.xDim, u0.yDim)


def mse_grad(u0, uT, upsample_factor=2):
    uT = uT.change_dim(upsample_factor*uT.xDim, upsample_factor*uT.yDim)
    u0 = u0.change_dim(upsample_factor*u0.xDim, upsample_factor*u0.yDim)
    a = angular_velocity_field()
    delta = get_sol_analytic(u0, a, 1.) - uT
    grad = get_sol_analytic(delta, lambda x, y: -a(x, y), 1.)
    return grad.change_dim(u0.xDim, u0.yDim)