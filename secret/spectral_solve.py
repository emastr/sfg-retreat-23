import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple

MAT = np.ndarray
   

def discretize(f: Callable[[MAT, MAT], MAT], n: int) -> MAT:
    """Evaluate the function f. f is a function of two variables,
     and is evaluated at n x n points uniformly spaced on [0,1]^2."""
    x = np.linspace(0, 1, n + 1)[:-1]
    y = np.linspace(0, 1, n + 1)[:-1]
    X, Y = np.meshgrid(x, y)
    return f(X, Y)


def fourier2D(f: MAT) -> MAT:
    """Compute the 2D Fourier transform of f."""
    return fft(fft(f, axis=0), axis=1) / f.shape[0] ** 2
    
    
def invFourier2D(coef: MAT) -> MAT:
    """Evaluate Fourier series with coefficients coef at points
    uniformly spaced on [0,1]^2. Result has same shape as coef."""
    return np.real(ifft(ifft(coef, axis=0), axis=1)) * coef.shape[0] ** 2


def get_freq(n: int) -> Tuple[MAT, MAT]:
    """Returns two n x n matrices Kx and Ky, with the frequencies
    for each fourier coefficient in a grid of size n x n."""
    freq = 2j * np.pi * fftfreq(n) * n
    return np.meshgrid(freq, freq)
    

def euler_solve(u0: MAT, L: MAT, b: Callable[[MAT], MAT], 
                N: int, T: float, callback: Callable=None) -> np.ndarray:
    """Solve the Matrix ODE u' - L*u = b(u) with initial condition u(0) = u0.
    u is a matrix, L is a matrix that is multiplied elementwise with u, b is a matrix valued function of u, 
    N is the number of times steps, T is the final time. Solution is returned as a matrix.
    """
    u = u0
    dt = T / N
    t = np.linspace(0, T, N)
    for n in range(1, N):
        u = euler_step(u, L, b, dt)
        if callback is not None:
            callback(u, n*dt, n)
    return u


def euler_step(u: MAT, L: MAT, b: Callable[[MAT], MAT], dt: float) -> MAT:
    """One step of the Euler Backwards method. u is a matrix,
    L is a matrix, b is a matrix valued function of u, dt is the step size."""
    return (u + dt * b(u)) / (1 - dt * L)


def get_L(Kx: MAT, Ky: MAT, lam: float) -> MAT:
    """Return the Laplace operator L*u = lam * (du/dx^2 + du/dy^2),
    represented as a matrix (frequency domain) with same dimensions as the fourier coefficients Kx and Ky.
    To evaluate the Laplace operator in the real domain, use invFourier2D(L * fourier2D(u))."""
    return lam * (Kx**2 + Ky**2)


def get_b(u_four: MAT, f_four: MAT, bx_real: MAT, by_real: MAT, Kx: MAT, Ky: MAT) -> MAT:
    """Return the fourier coefficients of f - (bx, by) dot grad(u),
     for a grid of size n x n. f_four is the fourier coefficients of f,
    bx_real and by_real are the real space values of bx and by, and 
    Kx and Ky are the fourier frequencies."""
    dxu_real = invFourier2D(Kx * u_four)
    dyu_real = invFourier2D(Ky * u_four)
    return f_four - fourier2D(bx_real * dxu_real + by_real * dyu_real)
    
    
def solve(u0: Callable[[MAT, MAT], MAT], f: Callable[[MAT, MAT], MAT], 
          bx: Callable[[MAT, MAT], MAT], by: Callable[[MAT, MAT], MAT], 
          lam: float, N: int, T: float, K: int, callback=None) -> MAT:
    """Solve the ODE u' - lam * laplace(u) + (bx, by) dot grad(u) = f
    with initial condition u(0) = u0. f is a function of u, T is the final time.
    N is the number of time steps and K is the number of Fourier bases.
    """
    
    bx_real = discretize(bx, K)
    by_real = discretize(by, K)
    Kx, Ky = get_freq(K)
    u0_four = fourier2D(discretize(u0, K))
    f_four = fourier2D(discretize(f, K))
    L_four = get_L(Kx, Ky, lam)
    b_four = lambda u: get_b(u, f_four, bx_real, by_real, Kx, Ky)
    u_freq  = euler_solve(u0_four, L_four, b_four, N, T, callback=callback)
    return invFourier2D(u_freq)


def solve_discrete(u0: MAT, bx: Callable[[MAT, MAT], MAT], by: Callable[[MAT, MAT], MAT], 
          lam: float, N: int, T: float, callback=None) -> MAT:
    """Solve the ODE u' - lam * laplace(u) + (bx, by) dot grad(u) = f
    with initial condition u(0) = u0. f is a function of u, T is the final time.
    N is the number of time steps and K is the number of Fourier bases.
    """
    K = u0.shape[0]
    assert u0.shape == (K, K), "u0_four must be a square matrix of fourier coefficients"
    
    bx_real = discretize(bx, K)
    by_real = discretize(by, K)
    Kx, Ky = get_freq(K)
    L_four = get_L(Kx, Ky, lam)
    u0_four = fourier2D(u0)
    b_four = lambda u: get_b(u, 0*u0_four, bx_real, by_real, Kx, Ky)
    u_freq  = euler_solve(u0_four, L_four, b_four, N, T, callback=callback)
    return invFourier2D(u_freq)


def plot_2d(f_list: List[MAT], **kwargs):
    """Plot a list of 2D functions. f_list is a list of 2D arrays.
     kwargs are passed to plt.imshow (plt.imshow([], **kwargs)). The functions are plotted in a grid."""
    n = int(np.ceil(np.sqrt(len(f_list))))
    for i,f in enumerate(f_list):
        plt.subplot(n, n, i + 1)
        plt.imshow(f, **kwargs)


def test_get_L():
    """Test the derivative of the Fourier transform."""
    ns = [2, 4, 8, 16, 32, 64, 128]
    er = []
    for n in ns:
        k = 2 * np.pi
        g = lambda x,y: np.exp(np.sin(k*x) + np.sin(k*y))
        Lg = lambda x,y: g(x,y) * k**2 * (np.cos(k*x)**2 - np.sin(k*x) + np.cos(k*y)**2 - np.sin(k*y))
        
        lam = 1.
        L = get_L(*get_freq(n), lam)
        g_disc = discretize(g, n)
        Lg_apx = invFourier2D(L * fourier2D(g_disc))
        
        er.append(np.linalg.norm(Lg_apx - discretize(Lg, n)) / n)
    
    plt.semilogy(ns, er, 'o-')
    plt.title('L2 error of Laplace operator')
    plt.xlabel('Number of basis functions')
    plt.ylabel('L2 error')
    

def test_get_b():
    k = 2 * np.pi
    f = lambda x,y: np.exp(np.sin(k*x) + np.sin(k*y))
    bx = lambda x,y: np.sin(k*x)
    by = lambda x,y: np.sin(k*y)
    u = lambda x,y: np.exp(np.sin(k*x) * np.sin(k*y))
    dxu = lambda x,y: k * np.cos(k*x) * np.sin(k*y) * u(x,y)	
    dyu = lambda x,y: k * np.sin(k*x) * np.cos(k*y) * u(x,y)
    b = lambda x,y: f(x,y) - (bx(x,y) * dxu(x,y) + by(x,y) * dyu(x,y))
    
    ns = [8, 16, 32, 64, 128]
    er = []
    
    for n in ns:    
        b_disc = discretize(b, n)
        b_apx = invFourier2D(get_b(fourier2D(discretize(u, n)), fourier2D(discretize(f, n)), 
                            discretize(bx, n), discretize(by, n), *get_freq(n)))
    
        er.append(np.linalg.norm(b_disc - b_apx) / n)
    plt.semilogy(ns, er, 'o-')
    

def test_euler():
    # Test the accuracy of the Euler method
    # By solving a simple ODE u' - (-u) = 0, u(0) = 1
    # Which has the solution u(t) = exp(-t).
    
    Ns = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512]).astype(float)
    er = []
    T = 1.
    for N in Ns:
        u = euler_solve(np.array([1.]), np.array([-1.]), lambda u: np.array([0.]), int(N), T)
        er.append(np.abs(u[0] - np.exp(-T)))
        
    plt.title("Test of Euler method for u' - (-u) = 0, u(0) = 1")
    plt.loglog(Ns, er, 'o-', label='error at time T=1')
    plt.plot(Ns, Ns**(-1), '--', label='O(N^-1)')
    plt.legend()


def stream_plot(bx, by):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y)
    bx = bx(x, y)
    by = by(x, y)
    plt.streamplot(x, y, bx, by, color=np.sqrt(bx**2 + by**2), cmap='Blues')