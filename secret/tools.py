from .function_tools import *
from .spectral_solve import *
import datetime


def get_u1(K=129):
    """Get the observed end state u1. This is a function of two variables.
    The type of object representing u1 has some nice properties. See the task 2 in the README.md.
    This object type will be returned by all the other functions."""
    return load_end_state(K=K)


def get_a():
    """Get the displacement field components. These are simply anonymous functions of x and y."""
    w = angular_velocity_field()
    ax = lambda x, y: w(x, y) * (y-0.5)   # 3 is nice setting for the test
    ay = lambda x, y: -w(x, y) * (x-0.5)
    return ax, ay


def solve_slow(u0, ax, ay, nu, N=100):
    """Solve the time dependent PDE given an initial condition u0,
    and evaluate the solution u1 at the final time T=1. 
    
    See task 2 in the README to understand the function object u0 and output u1.
    
    You can specify the diffusivity nu as a float, and the velocity field ax, ay as functions of x, y."""
    u0 = u0.eval_grid()
    callback = bar_callback(N, N//10)
    uT = solve_discrete(u0, lambda x,y: -ax(x, y), lambda x,y: -ay(x,y), lam=nu, N=N, T=1., callback=callback)
    return to_basis(uT)


def solve_fast(u0, upsample_factor=2):
    """
    Solve the time dependent PDE given an initial condition u0,
    and evaluate the solution u1 at the final time T=1. 
    
    See task 2 in the README to understand the function object u0 and output u1.
    """
    u0 = u0.change_dim(upsample_factor*u0.xDim, upsample_factor*u0.yDim)
    u1 = get_sol_analytic(u0, angular_velocity_field(), 1.)
    return u1.change_dim(u0.xDim, u0.yDim)


def mse_grad(u0, u1, upsample_factor=2):
    """"Compute gradient of mse with respect to u0.
    
    See task 2 in the README to understand the function objects u0 and u1.    
    
    u0 is a function of two variables
    u1 is a function of two variables
    The returned object is of the same type as u0 and u1
    upsample_factor affects the accuracy of the gradient. factor >= 2 is recommended."""
    u1 = u1.change_dim(upsample_factor*u1.xDim, upsample_factor*u1.yDim)
    u0 = u0.change_dim(upsample_factor*u0.xDim, upsample_factor*u0.yDim)
    a = angular_velocity_field()
    delta = get_sol_analytic(u0, a, 1.) - u1
    grad = get_sol_analytic(delta, lambda x, y: -a(x, y), 1.)
    return grad.change_dim(u0.xDim, u0.yDim)


def decrypt(guess, cheat=False):
    january1st = datetime.datetime(2023, 1, 1)
    timesince = datetime.datetime.now() - january1st
    T = int(timesince.total_seconds() / 60)

    vals = [152,            # 60 phd students + 40 prof + 30 postdoc + 10 staff = 140
            10554692,       # 10 million +- 1 million
            0.9772499,      # symmetric 2sigma ~ 95 so symmetric 97.5+-0.5. # Chebyshev: 1-1/4 /2 = 1-1/8 = 0.875
            149,            # ~20/30 in stockholm, 2 million in stockholm so 100 total
            0.43,           # 430 ms. 
            1220000,        # 1.2 million (i got 1.4)
            22.4591577184,  # Pi to e ~ 3^3 =  
            112,            # Know 54 in double, double it so get 108.
            T,
            11669007708,   # 1 square m. coast line (): 20000 km. 
            0.5073,
            282.8427]
    
    # Takes 6 hours across sweden at 100 km/h. 600+-100 km across sweden.
    # take 3+-1:1 ratio on sweden length, so get coast line of around 600*5 = 3000 km +- 500 km.
    # Take 1 km deep archipelago, out of maybe 40% is islands we get
    # 3000*0.4*1 = 3000 km^2. -> 3000 * 10^6 = 3 * 10^9 yoga mats
    # 
    # P(no same ) = (365*364*363 * .... *(365-23))/365^23 ~ 23 * 23/365 ~
    # 2 sqrt(20 000) = 2 * sqrt(2 * 10^4) = 2*sqrt(2)*10^2 = 2*1.1414*100 = 282.84
    # 2 * 141 = 282
    
    good_guess = [(130, 170), (1e7, 1.1e7), (0.97, 0.98), 
                  (100, 200), (0.2, 0.5), (5e4, 5e6), 
                  (21, 24), (105, 115), (T-2, T+2), 
                  (2e9, 2e10), (0.4, 0.6), (282.83, 282.85)]
    
    bad_guess = [(100, 200), (9e6, 1.1e7), (0.85, 1.0), 
                 (50, 250), (0.1, 1.), (1e4, 1e7), 
                 (25, 30), (64, 128), (T-60, T+60),
                 (1e8, 1e10), (0.1, 0.6), (281, 283)]
    
    is_in = lambda interval, x: interval[0] <= x <= interval[1]
    
    def score(guess):
        penalties = [g[1]/g[0] for v, g in zip(vals, guess) if is_in(g, v)]
        return (10 + sum(penalties))*2**(12 - len(penalties))
        
    
    score_bad = score(bad_guess)
    score_good = score(good_guess)
    score_guess = score(guess)
    
    isin = ['Y' if is_in(g,v) else 'N' for v, g in zip(vals, guess)]
    isin = ', '.join([str(i+1) + ':' + v for i, v in enumerate(isin)])
    
    if cheat:
        score_guess = 0
    
    messages = [
        "Score > 5000. You successfully decrypted A: from secret.tools import get_a.",
        "Score < 5000. You successfully decrypted B: from secret.tools import solve_slow.",
        "Score <  150. You successfully decrypted C: from secret.tools import solve_fast.",
        "Score <  100. You successfully decrypted D: from secret.tools import mse_grad.",
    ]
    
    thresholds = [5000, 250, 130]
    unlocks  = [0] + [i+1  for i, t in enumerate(thresholds) if score_guess <= t]
    
    print("======  STATUS REPORT:   ======", end="\n")
    print(f"Your score was {score_guess}.", end="\n")
    print(f"A good score is {score_good:.0f}. A bad score is {score_bad:.0f}", end="\n")
    print(f"Overlapping with itervals: {isin}", end="\n")
    
    print("======  DECRYPTED FUNCTIONS:   ======", end="\n")
    for u in unlocks:
        print(messages[u], end="\n")

    print("Use help() command if the README.md is not clear enough.")
