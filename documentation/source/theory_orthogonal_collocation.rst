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

   \dot{x} &= f(x(t),u(t),z(t),p_{tv}(t),p(t)),\\\\
   0 &= g(x(t),u(t),z(t),p_{tv}(t),p(t)).\\\\

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
        OC [label="orthogonal collocation \n on finite elements", fillcolor="#edf0f2", style="filled"]



        ODE -> direct, indirect
        indirect -> PMP
        direct -> simultaneous, sequential

        sequential -> single_shoot, full_disc
        simultaneous -> mult_shoot, full_disc
        full_disc -> Euler, RungeKutta, OC
    }

**do-mpc** is based on **orthogonal collocation on finite elements** which is a direct, simultaneous, full discretization approach.

* **Direct**: The continuous time variables are discretized to transform the infinite-dimensional optimal control problem to a finite dimensional nonlinear programming (NLP) problem.

* **Simultaneous**: Both the control inputs and the states are discretized.

* **Full discretization**: A discretization scheme is hand implemented in terms of symbolic variables instead of using an ODE/DAE solver.

The full discretization is realized with **orthogonal collocation on finite elements** which is discussed in the remainder of this post.
The content is based on [Biegler2010]_.




Lagrange polynomials for ODEs
*****************************
To simplify things, we now consider the following ODE:

.. math::

    \dot{x} = f(x), \quad x(0)=x_0,

Fundamental for orthogonal collocation is the idea that the solution of the ODE
:math:`x(t)` can be approximated accurately with a polynomial of order :math:`K+1`:

.. math::

    x^K_i(t) = \alpha_0 + \alpha_1 t + \dots + \alpha_{K}  t^K.

This approximation should be valid on small time-intervals :math:`t\in [t_i, t_{i+1}]`, which
are the **finite elements** mentioned in the title.

The interpolation is based on :math:`j=0,\dots,K` interpolation points :math:`(t_j, x_{i,j})` in the interval :math:`[t_i, t_{i+1}]`.
We are using the **Lagrange interpolation polynomial**:


.. math::

    &x^K_i(t) = \sum_{j=0}^K L_j(\tau) x_{i,j}\\
    \text{where:}\quad
    &L_j(\tau) = \prod_{
    \begin{array}{c}k=0\\ k \neq j \end{array}
    }^K \frac{(\tau-\tau_k)}{(\tau_j-\tau_k)}, \quad \tau &= \frac{t-t_i}{\Delta t_i}, \quad \Delta t_i=t_{i+1}-t_i.


We call :math:`L_j(\tau)` the Lagrangrian basis polynomial with the dimensionless time :math:`\tau \in [0,1]`.
Note that the basis polynomial :math:`L_j(\tau)` is constructed to be :math:`L_j(\tau_j)=1` and :math:`L_j(\tau_i)=0`
for all other interpolation points :math:`i\neq j`.

This polynomial ensures that for the interpolation points :math:`x^K(t_{i,j})=x_{i,j}`.
Such a polynomial is fitted to all finite elements, as shown in the figure below.

.. _my-reference-label:
.. figure:: static/orthogonal_collocation.svg

    Lagrange polynomials representing the solution of an ODE on neighboring finite elements.

Note that the collocation points (round circles above) can be choosen freely
while obeying :math:`\tau_0 = 0` and :math:`\tau_{j}<\tau_{j+1}\leq1`.
There are, however, better choices than others which will be discussed in :ref:`secOrthogonalPoly`.

Deriving the integration equations
**********************************

So far we have seen how to approximate an ODE solution
with Lagrange polynomials **given a set of values from the solution**.
This may seem confusing because we are looking for these values in the first place.
However, it still helps us because we can now state conditions based on this polynomial representation
that **must hold for the desired solution**:

.. math::

    \left.\frac{d x^K_i}{dt}\right|_{t_{i,k}} = f(x_{i,k}), \quad k=1,\dots,K.

This means that the time derivatives from our polynomial approximation evaluated
**at the collocation points** must be equal to the original ODE at these same points.

Because we assumed a polynomial structure of :math:`x^K_i(t)` the time derivative can be conveniently expressed as:

.. math::

    \left.\frac{d x^K_i}{dt}\right|_{t_{i,k}} = \sum_{j=0}^K \frac{x_{i,j}}{\Delta t}
    \underbrace{\left.\frac{d L_j}{d \tau}\right|_{\tau_k}}_{a_{j,k}},

for which we substituted :math:`t` with :math:`\tau`.
It is important to notice that **for fixed collocation points** the terms :math:`a_{j,k}`
are constants that can be pre-computed.
The choice of these points is significant and will be discussed in
:ref:`secOrthogonalPoly`.

Collocation constraints
=======================

The solution of the ODE, i.e. the values of :math:`x_{i,j}` are now obtained by solving
the following equations:

.. math::

    \sum_{j=0}^K a_{j,k} \frac{x_{i,j}}{\Delta t} = f(x_{i,k}), \quad k=1,\dots,K.

Continuity constraints
======================

The avid reader will have noticed that through the collocation constraints
we obtain a system of :math:`K-1` equations for :math:`K` variables, which is insufficient.

The missing equation is used to ensure continuity between the finite elements shown in the figure above.
We simply enforce equality between the final state of element :math:`i`, which we denote :math:`x_i^f`
and the initial state of the successive interval :math:`x_{i+1,0}`:

.. math::

    x_{i+1,0} = x_{i}^f

However, with our choice of collocation points :math:`\tau_0=0,\ \tau_j<\tau_{j+1}\leq 1,\ j=0,\dots,K-1`,
we do not explicitly know :math:`x_i^f` in the general case (unless :math:`\tau_{K} = 1`).

We thus evaluate the interpolation polynomial again and obtain:

.. math::

    x_i^f = x^K_i(t_{i+1}) = \sum_{j=0}^K \underbrace{L_j(\tau=1)}_{d_j} x_{i,j},

where similarly to the collocation coefficients :math:`a_{j,k}`, the continuity coefficient :math:`d_j` can be precomputed.

Solving the ODE problem
=======================

It is important to note that orthogonal collocation on finite elements is an **implict ODE integration scheme**, since we need
to evaluate the ODE equation for yet to be determined future states of the system.
While this seems inconvenient for simulation, it is straightforward to incorporate in a
model predictive control (MPC) or moving horizon estimation (MHE) formulation, which are
essentially large constrained optimization problems of the form:

.. math::

    \min_z \quad &c(z)\\
    \text{s.t.:} \quad & h(z) = 0\\
    & g(z) \leq 0

where :math:`z` now denotes a generic optimization variable,
:math:`c(z)` a generic cost function and :math:`h(z)` and :math:`g(z)` the equality and inequality constraints.

Clearly, the equality constraints :math:`h(z)` can be extended with the above mentioned collocation constraints,
where the states :math:`x_{i,j}` are then optimization variables of the problem.

Solving the MPC / MHE optimization problem then implictly calculates the solution of the governing ODE
which can be taken into consideration for cost, constraints etc.


.. _secOrthogonalPoly:

Collocation with orthogonal polynomials
=======================================

Finally we need to discuss how to choose the collocation points :math:`\tau_j,\  j=0,\dots, K`.
Only for fixed values of the collocation points the collocation constraints become mere algebraic equations.

**Just a short disclaimer**:
Ideal values for the collocation points are typically found in tables, e.g. in [Biegler2010]_.
The following simply illustrates how these suggested values are derived and are not implemented in practice.

We recall that the solution of the ODE can also be determined with:

.. math::

    x(t_i) = x(t_{i-1}) + \int_{t_{i-1}}^{t_i} f(x(t)) dt,

which is solved numerically with the quadrature formula:

.. math::

    &x(t_i) = x(t_{i-1}) + \sum_{j=1}^K \omega_j  \Delta t f(x(t_{i,j})\\
    &t_{i,j} = t_{i-1} + \tau_j \Delta t

The collocation points are now chosen such that the quadrature formula provides an
exact solution for the original ODE if :math:`f(x(t)` is a polynomial in :math:`t` of order :math:`2K`.
It shows that this is achieved by choosing :math:`\tau` as the roots of a :math:`k`-th degree polynomial :math:`P_K(\tau)`
which fulfils the **orthogonal property**:

.. math::

    \int_0^1 P_i(\tau) P_{j}(\tau) = 0, \quad i=0,\dots, K-1,\ j=1,\dots, K

The resulting collocation points are called **Legendre roots**.

Similarly one can compute collocation points from the more general **Gauss-Jacoby** polynomial:

.. math::

    \int_0^1 (1-\tau)^{\alpha} \tau^{\beta} P_i(\tau) P_{j}(\tau) = 0, \quad i=0,\dots, K-1,\ j=1,\dots, K

which for :math:`\alpha=0,\ \beta=0` results exactly in the Legrendre polynomial from above
where the truncation error is found to be :math:`\mathcal{O}(\Delta t^{2K})`.
For :math:`\alpha=1,\ \beta=0` one can determine the **Gauss-Radau** collocation points with truncation error
:math:`\mathcal{O}(\Delta t^{2K-1})`.

Both, Gauss-Radau and Legrende roots are commonly used for orthogonal collocation and can be selected
in **do-mpc**.


For more details about the procedure and the numerical values for the collocation points we refer to [Biegler2010]_.


Bibliography
************

.. [Biegler2010] L.T. Biegler. Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes. SIAM, 2010.
