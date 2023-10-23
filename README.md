##  sfg-retreat-23. Ground rules:
* ``git clone git@github.com:emastr/sfg-retreat-23.git``
* Be nice :) 
* Don't look in the ``secret`` folder unless you're given access to a certain function or package.
* Install  the packages ``matplotlib, numpy, typing, datetime``.

## Teams
1) 
2) 
3) 
4) 
5) 

# Intro: The Message. 

You realize that one of the Postdocs didn't attended your group meeting today. On the way to check if they are at their desk, 
you spot a mysterious figure exit their office. The figure disappears before you can have a good look at their face. 
Entering the office, you see that it is empty, with no sign of the postdoc. On their desk is a cup of coffee, cold to the touch. On their whiteboard is a message:

![jpg](secret/end.jpg)

You notice that the message was written with a permanent marker. This message was important. Maybe a ground-breaking discovery? 
The person who tried to erase the message must have gotten surprised by your arrival, because instead of erasing the message, 
they smudged it. You realize that uncovering this message might be crucial in figuring out why our Postdoc has gone missing. 

## Main Task
Your task is to recover the smudged message.

## Analysis
After talking to our colleagues at the physics department, you find out that the smudge has been carried out with surgical precision.
In fact, if we consider the whiteboard to be a unit square $[0,1]^2$ and consider the concentration of ink $u(t, x,y)$ at time $t$
since the perpetrator started smudging, one can model the smudging process as a PDE:
$$\partial_t u(t,\pmb x) - a(\pmb x)\cdot \nabla u(t,\pmb x) = \nu \Delta u(t,\pmb x), \quad \pmb x \in [0,1]^2 \quad \text{and} \quad t\in [0,1],$$
where $a(x,y)=(a_x(x,y), a_y(x,y))$ is the displacement vector field, and $\nu = 0.00004$ the diffusivity. 
The initial condition is given by the original message $u_0(x,y)$, and is precisely the quantity that we wish to estimate:
$$u(0, x, y) = u_0(x,y),\quad x,y\in [0,1]^2.$$
The image $u_1$ we observed is at the end state, $u(1, x, y) = u_1(x,y)$. We can assume that $u$ is periodic on the unit square. 

## Tools
At your disposal you have a trusty set of tools for solving this problem. 
However, upon returning to your office, you notice that someone has hacked into the computer and ENCRYPTED all of your code! 
In order to obtain these tools, you first need to decrypt the secret folder. To do this, you have to answer a set of nefariously
constructed questions. It is clear that someone REALLY doesn't want you to figure out what the smudged message says. 

## First Task

Read the pdf ``KTH-Estimathon.pdf``. To uncover the encrypted tools, you need to call the function ``decrypt``. 
You can do this in the notebook ``task-1.ipynb``. The function takes as input a list of tuples ``[(min_1, max_1), (min_2, max_2), (), ...]`` 
with the upper and lower bounds that you arrived at, and unlocks your tools if you were sufficiently close. Depending on your score,
you can get the following tools:
* A - Score above 5000: Returns the two functions ``ax`` and ``ay``. The input for these functions must be two equally sized numpy arrays ``x`` and ``y``. The output will be a vector corresponding to $a_x($``x``,``y``$)$ and $a_y($``x``,``y``$)$, respectively.
* B - Score 5000 or lower: Returns the end state of the solution to our PDE, with given choices of ``u0``, ``a`` and $\nu$. Takes also as input the number of time steps ``N``, which if set too low might make the solution unreliable or even unstable.
* C - Score 150 or lower: this fast solver lets you obtain the end state $u_1$ of the PDE without diffusion ($\nu=0$), but doesn't let you change the velocity field ``a``. It takes as input ``u0``.
* D - Score 100 or lower: This incredible function returns the function space gradient of the mean square error between the smudged message $u_1$, and the final state $\tilde u_1(\tilde u_0)$ that corresponds to an approximation $\tilde u_0$ of the original message. In other words, this function returns the gradient $\nabla J(\tilde u_0)$ if we $J$ is defined as  
$$J(\tilde u_0) = \int_0^1\int_0^1 (u_1-\tilde u_1(\tilde u_0))^2\mathrm{d}x\mathrm{d}y.$$
#### You can keep trying multiple times, but please report the first guess to us. 

## Second Task
Load in the smudged message as a function. For this, you can use ``from secret.tools import get_u1``. Calling ``u1 = get_u1()`` will return a function object. It has multiple functionalities:
* You can plot using ``u1.plot(plt.gca(), cmap='gray')`` (after ``import matplotlib.pyplot as plt``).
* You can add two of these functions: ``u1 + u2`` returns an object of the same type. Same goes for multiplication with scalars, i.e. ``2.*u1``
* You can take derivatives: ``u.diff(degree_x, degree_y)``.
* You can evaluate in vectors ``x``, ``y`` by calling ``u1(x, y)``. This is slow.
* If your score was too high, you might need to write your own solver. You can extract the function values of $u_1$ on a uniform grid by calling ``u1.eval_grid()``. This operation is fast.

Play around with the object for a couple minutes in the notebook ``task-2.ipynb``. Try for example ``(u1.diff(0,1)**2 + u1.diff(1, 0)**2).plot(plt.gca())`` 

 ## Third Task
 Uncover the hidden message, and explain what happened to the missing postdoc. Use the notebook ``task-3.ipynb``. Write your best explanation and hand it in to Emanuel or Pip.






