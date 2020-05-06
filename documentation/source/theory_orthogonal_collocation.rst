*****************************************
Orthogonal collocation on finite elements
*****************************************

A **dynamic system model** is at the core of all model predictive control (MPC) and moving horizon estimation (MHE)
formulations.
This model allows to predict and optimize the future behavior of the system (MPC)
or establishes the relationship between past measurements and estimated states (MHE).

When working with **do-mpc** an essential question is whether a
**discrete** or **continuous** model is supplied.
The discrete time formulation:

.. math::

   x_{k+1} &= f(x_{k},u_{k},z_{k},p_{tv,k},p),\\\\
   0 &= g(x_{k},u_{k}, z_{k},p_{tv,k},p),\\\\


gives an explicit relationship for the future states :math:`x_{k+1}`
based on the current states :math:`x_k`, inputs :math:`u_k`,
algebraic states :math:`z_k` and further parameters :math:`p`, :math:`p_{tv,k}`.
It can be evaluated in a straight-forward fashion to recursively obtain the future states of the system,
based on an initial state :math:`x_0` and a sequence of inputs.

However, many dynamic model equations are given in the continuous time form as ordinary differential equations (ODE)
or differential algebraic equations (DAE):

.. math::

   \dot{x} &= f(x,u,z,p_{tv},p),\\\\
   0 &= g(x,u,z,p_{tv},p).\\\\

Incorporating the ODE/DAE is typically less straight-forward than their discrete-time counterparts and a variety of methods are applicable.
An (incomplete!) overview and classification of commonly used methods is shown in the diagram below:

.. graphviz::
    :name: ode_dae_MPC
    :caption: Approaching an ODE/DAE continuous model for MPC or MHE.
    :align: center

    digraph G {
        node [shape="rectangle"];
        ODE [label="ODE/DAE continous time model"]
        direct
        indirect
        PMP [label="Pontryagin minimum principle"]
        sequential
        simultaneous [label="simultaneous"]
        single_shoot [label="single shooting"]
        mult_shoot [label="multiple shooting"]
        full_disc [label="full discretization"]
        Euler
        RungeKutta [label="Runge-Kutta"]
        OC [label="orthogonal collocation \n on finte elements", fillcolor="#edf0f2", style="filled"]



        ODE -> direct, indirect
        indirect -> PMP
        direct -> simultaneous, sequential

        sequential -> single_shoot
        simultaneous -> mult_shoot, full_disc
        full_disc -> Euler, RungeKutta, OC
    }

**do-mpc** is based on **orthogonal collocation on finite elements** which is a direct, simultaneous, full discretization approach.

    **Direct**: The continous time variables are discretized to transform the infinite-dimensional optimal control problem
    to a finite dimensional nonlinear programming (NLP) problem.

    **Simultaneous**: Both the control inputs and the states are discretized.

    **Full discretization**: A discretization scheme is hand implemented in terms of symbolic variables instead of using an ODE/DAE solver.

The full discretization is realized with **orthogonal collocation on finite elements** which is discussed in the remainder of this post.
The content is based on [Biegler2010]_.




Lagrange polynomials for ODEs
*****************************
To simplify things, we now consider the following ODE:

.. math::

    \dot{x} = f(x), \quad x(0)=x_0,

which we assume can be approximated accurately with a polynomial of order :math:`K+1`:

.. math::

    x^K(t) = \alpha_0 + \alpha_1 t + \dots + \alpha_{K}  t^K.

This approximation should be valid on small time-intervals :math:`t\in [t_i, t_{i+1}]`.
With :math:`j=0,\dots,K` interpolation points :math:`x_{i,j}` in the interval,
we can use **Lagrange interpolation polynomials** to obtain:

.. math::


    &x^K(t) = \sum_{j=0}^K L_j(\tau) x_{i,j}\\
    \text{where:}\quad
    &L_j(\tau) = \prod_{
    \begin{array}{c}k=0\\ k \neq j \end{array}
    }^K \frac{(\tau-\tau_k)}{(\tau_j-\tau_k)}, \quad \tau &= \frac{t-t_i}{\Delta t_i}, \quad \Delta t_i=t_{i+1}-t_i.

This polynomial ensures that for the interpolation points :math:`x^K(t_{i,j})=x_{i,j}`.


Bibliography
************

.. [Biegler2010] L.T. Biegler. Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes. SIAM, 2010.
