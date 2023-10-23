from .basis_scaled import *
import matplotlib.image as im
import os
from typing import List, Callable, Tuple

MAT = np.ndarray

def to_basis(U: MAT, K=None):
    coef = BasisProduct._interpolate(U, FourBasis, FourBasis)    
    basis = BasisProduct(coef, coef.shape[0], coef.shape[1], FourBasis, FourBasis)
    if K is not None:        
        basis = basis.change_dim(K, K)
    return basis

def angular_velocity_field():
    r = lambda x,y: np.sqrt((x-0.5)**2 + (y-0.5)**2)
    sigmoid = lambda t: 1/(1+np.exp(-t))
    w = lambda x, y: (1-sigmoid((r(x,y)-0.2)/0.05)) * (1. + 0.7*np.cos(9*np.pi*r(x,y))) / 0.7
    return w

def get_sol_analytic(u0: MAT, w: Callable[[MAT, MAT], MAT], T) -> Callable[[MAT, MAT], MAT]:
    """ Returns a function uT(x,y) that is the solution of the ODE
    u' - a(x,y) * (du/dx, du/dy) = 0, u(0) = u0(x,y) at time T.
    u0 is a function of two variables, a is a function of two variables,
    and T is a scalar.
    
    Args:
        u0 MAT: _description_
        a (Callable[[MAT, MAT], MAT]): _description_
        T (_type_): _description_

    Returns:
        MAT: _description_
    """
    X, Y = u0.grid()
    U = u0.eval_grid()
    
    N = X.shape[0]
    x = X.flatten()
    y = Y.flatten()
    theta = w(x, y) * T
    xT = ((x-0.5) * np.cos(theta) - (y-0.5) * np.sin(theta)+0.5)%1.
    yT = ((x-0.5) * np.sin(theta) + (y-0.5) * np.cos(theta)+0.5)%1.
    
    idx_x = np.floor(xT * N).astype(int)
    idx_y = np.floor(yT * N).astype(int)
    U0 = U[idx_x, idx_y].reshape(N, N)
    return to_basis(U0)
    
    
    
    
    
    def uT(x, y):
        theta = w(x, y) * T
        return u0((x-0.5) * np.cos(theta) - (y-0.5) * np.sin(theta)+0.5, 
                  (x-0.5) * np.sin(theta) + (y-0.5) * np.cos(theta)+0.5)
    return uT

def get_mse_gradient_analytic(uT: Callable[[MAT, MAT], MAT],
                     a: Callable[[MAT, MAT], MAT], 
                     T: float) -> Callable:
    """Returns the gradient of the mean squared error of the solution
    of the ODE u' - a(x,y) * (du/dx, du/dy) = 0, u(0) = u0(x,y) at time T.
    u0 is a function of two variables, a is a function of two variables,
    and T is a scalar.
    """
    def grad(u0: Callable[[MAT, MAT], MAT]) -> Callable[[MAT, MAT], MAT]:
        sol_u0 = get_sol_analytic(u0, a, T)
        delta = lambda x, y: sol_u0(x, y) - uT(x, y)
        sol_delta = get_sol_analytic(delta, lambda x, y: -a(x, y), T)
        return sol_delta
    
    return grad     

def state_to_grid(state):
    return state.eval_grid()

def load_end_state(K=129):
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, 'end.npy')
    
    # Load image
    img = np.load(file_path)
    basis = to_basis(img, K)
    return basis

def load_init_state(K=129):    
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, 'truth.jpg')

    # Load image
    img = np.sum(im.imread(file_path), axis=2).astype(float)/255
    img = img[:, :]

    coef = BasisProduct._interpolate(img[::-1, :].T, FourBasis, FourBasis)

    # Custom blur
    blur = 0.007
    s = lambda x,y: np.exp(-((x-0.)**2 + (y-0.)**2)/blur**2)
    coef = coef * BasisProduct._interpolate_func(s, coef.shape[0], coef.shape[1], FourBasis, FourBasis)

    
    basis = BasisProduct(coef, coef.shape[0], coef.shape[1], FourBasis, FourBasis)
    basis = basis.change_dim(K, K)
    return basis