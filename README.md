# The Message. sfg-retreat-23 

You realize that one of the Postdocs didn't attended the group meeting today. On the way to check if they are at their desk, 
you spot a mysterious figure exit their office. The figure disappears before you can have a good look at their face. 
Entering the office, you see that it is empty, with no sign of the postdoc. On their whiteboard is a message:



You notice that the message was written with a permanent marker. This message was important. Maybe a ground-breaking discovery? 
The person who tried to erase the message must have gotten surprised by your arrival, because instead of erasing the message, 
they smudged it. You realize that uncovering this message might be crucial in figuring out why our Postdoc has gone missing. 

## Task:
Your task is to recover the smudged message.

## Tools:
After talking to our colleagues at the physics department, we find out that the smudge has been carried out with surgical precision.
In fact, if we consider the whiteboard to be a unit square $[0,1]^2$ and consider the concentration of ink $u(t, x,y)$ at time $t$
since the perpetrator started smudging, one can model the smudging process as a PDE:
$$
    \partial_t u - a\cdot \nabla u = \nu \Delta u, \quad x,y\in [0,1]^2 \quad \text{and} \quad t>0,
$$
with the initial condition given by the original message $u_0(x,y)$:
$$
    u(0, x, y) = u_0(x,y),\quad x,y\in [0,1]^2.
$$
We can assume that $u$ is periodic on the unit square.
