**********************************
Basics of model predictive control
**********************************

**Model predictive control (MPC)** is a control scheme where a model is used for predicting the behavior of the system for a finite time window, the horizon.
Based on these predictions and the current measured/estimated state of the system, the optimal control inputs with respect to a defined control objective and subject to system constraints is computed.
After a certain time interval, the measurement, estimation and computation process is repeated with a shifted horizon.
This is the reason why this method is also called **receding horizon control (RHC)**.

.. image:: anim.gif

The MPC principle is visualized in the graphic above.
The dotted line indicates the current prediction and the solid line represents the realized values.
The graphic is generated using the innate plotting capabilities of **do-mpc**.

In the following, we will present the type of models, we can consider.
Afterwards, the (basic) **optimal control problem (OCP)** is presented.
Finally, **multi-stage NMPC**, the approach for robust NMPC used in **do-mpc** is explained.

System model
============

The system model plays a central role in MPC.
**do-mpc** enables the optimal control of continuous and discrete-time nonlinear and uncertain systems.
For the continuous case, the system model is defined by

.. math::

    \dot{x}(t) = f(x(t),u(t),z(t),p(t),p_{\text{tv}}(t)), \\
    y(t) = h(x(t),u(t),z(t),p(t),p_{\text{tv}}(t)),

and for the discrete-time case by

.. math::

    x_{k+1} = f(x_k,u_k,z_k,p_k,p_{\text{tv},k}), \\
    y_k = h(x_k,u_k,z_k,p_k,p_{\text{tv},k}).

The state of the systems are given by :math:`x(t),x_k`, the control inputs by :math:`u(t),u_k`, algebraic states by :math:`z(t),z_k`, (uncertain) parameters by :math:`p(t),p_k`, time-varying (but known) parameters by :math:`p_{\text{tv}}(t),p_{\text{tv},k}` and measurements by :math:`y(t),y_k`, respectively.
The time is denoted as :math:`t` for the continuous system and the time steps for the discrete system are indicated by :math:`k`.


Model predictive control problem
================================

For the case of continuous systems, trying to solve OCP directly is mostly computationally intractable because it is an infinite-dimensional problem.
**do-mpc** uses a full discretization method, namely `orthogonal collocation`_, to discretize the OCP and to allow the find a solution with state-of-the-art numerical solvers.
This means, that both the OPC for the continuous and the discrete system result in a similar discrete OPC.

.. _`orthogonal collocation`: theory_orthogonal_collocation.html

For the application of MPC, the current state of the system needs to be known.
In general, the measurement :math:`y_k` does not contain the whole state vector, which means a state estimate :math:`\hat{x}_k` needs to be computed.
The state estimate can be derived e.g. via `moving horizon estimation`_.

.. _`moving horizon estimation`: theory_mhe.html

The OCP is then given by:

.. math::

    &\min_{\mathbf{x}_{0:N},\mathbf{u}_{0:N-1},\mathbf{z}_{0:N}} & & m(x_N,z_N,p_N,p_{\text{tv},N}) + \sum_{k=0}^{N-1} l(x_k,z_k,u_k,p_k,p_{\text{tv},k}) && \\
    &\text{subject to} & & x_0 = \hat{x}_0, & \\
    &&& x_{k+1} = f(x_k,u_k,p_k,p_{\text{tv},k}), &\, \forall k=0,\dots,N-1,\\
    &&& g(x_k,u_k,p_k,p_{\text{tv},k}) \leq 0 &\, \forall k=0,\dots,N-1, \\
    &&& x_{\text{lb}} \leq x_k \leq x_{\text{ub}}, &\, \forall k=0,\dots,N-1, \\
    &&& u_{\text{lb}} \leq u_k \leq u_{\text{ub}}, &\, \forall k=0,\dots,N-1, \\
    &&& z_{\text{lb}} \leq z_k \leq z_{\text{ub}}, &\, \forall k=0,\dots,N-1, \\
    &&& g_{\text{terminal}}(x_N,z_N) \leq 0, &

where :math:`N` is the prediction horizon and :math:`\hat{x}_0` is the current state estimate, which can be computed based on past measurements :math:`y_k`, inputs :math:`u_k` and state estimates :math:`\hat{x}_k`, :math:`k < 0`.
One optimization based method to do this is `moving horizon estimation`_.
**do-mpc** allows to set upper and lower bounds for the states :math:`x_{\text{lb}}, x_{\text{ub}}`, inputs :math:`u_{\text{lb}}, u_{\text{ub}}` and algebraic states :math:`z_{\text{lb}}, z_{\text{ub}}`.
Terminal constraints can be enforced via :math:`g_{\text{terminal}}(\cdot)` and general nonlinear constraints can be defined with :math:`g(\cdot)`, which can also be realized as soft constraints.
The objective function consists of two parts, the mayer term :math:`m(\cdot)` which gives the cost of the terminal state and the lagrange term :math:`l(\cdot)` which is the cost of each stage :math:`k`.

.. _`moving horizon estimation`: theory_mhe.html

This formulation is the basic formulation of the OCP, which is solved by **do-mpc**.
In the next section, we will explain how **do-mpc** considers uncertainty to enable robust control.

.. note::
    Please be aware, that due to the discretization in case of continuous systems, a feasible solution only means that the constraints are satisfied point-wise in time, which in general provides very good results.


Robust multi-stage NMPC
=======================

The basic idea for the multi-stage approach is to consider various scenarios, where a scenario is defined by one possible realization of all uncertain parameters.
The family of all considered discrete scenarios can be represented as a tree structure, called the scenario tree:

.. image:: scenario_tree.png

where one scenario is one path from the root node on the left side to one leaf node on the right.
At every instant, the MPC problem at the root node is solved while explicitly taking into account the uncertain future evolution and the existence of future decisions, which can exploit the information gained throughout the evolution progress along the branches.
Through this design, feedback information is considered in the open-loop optimization problem, which reduces the conservativeness of the multi-stage approach.
Considering feedback information also means, that decisions :math:`u` branching from the same node need to be identical, because they are based on the same information (non-antipacivity constraints).

The system equation for a discretized/discrete system in the mutli-stage setting is given by:

.. math::

    x_{k+1}^j = f(x_k^{p(j)},u_k^j,z_k^{p(j)},p_k^{r(j)},p_{\text{tv},k}),

where the function :math:`p(j)`` refers to the parent state :math:`x_k^{p(j)}` and the considered realization of the uncertainty is given by :math:`r(j)` via :math:`d_k^{r(j)}`

In this exemplary tree, the control problem contains one uncertain parameters :math:`p`, for which 3 explicit values are considered.
Due to the tree structure of the multi-stage approach, the number of scenarios grows exponentially with the prediction horizon :math:`N`.
To reduce the computational load, the robust horizon :math:`N_{\text{robust}} \leq N` is introduced, after which the branching stops.
Hence, the number of considered scenarios is given by:

.. math::

    N_{\text{s}} = (\prod_{i=1}^{n_p} n_{v,i})^{N_{\text{robust}}}

where :math:`n_p` is the number of parameters and :math:`n_{v,i}` is the number of explicit values considered for the :math:`i`-th parameter.
This results in :math:`N_{\text{s}} = 27` scenarios for the presented scenario tree above.

The collection of child nodes
Explain scenario tree. Parent nodes, etc.
The constraints need to satisfied for all possible combinations of the constraints.
Each parent node branches :math:`\prod_{1}^{n_p} n_{v,i}` times within the robust horizon :math:`N_{\text{robust}}`.
For the last :math:`N-N_{\text{robust}}` steps the values of the uncertainties are kept constant.
This requirement is included in the formulation of the optimal control problem:

.. math::

    & \min_{\mathbf{x}_{0:N}} &&\, \tilde{J} & \\
    &\text{subjet to} & & \, x_0 = \hat{x}_0 & \\
    &&& \, x_{k+1}^j = f(x_k^{p(j)},u_k^j,z_k^{p(j)},p_k^{r(j)},p_{\text{tv},k}) & \, \forall \\
    &&& u_k^i = u_k^j \text{ if }  x_k^{p(i)} = x_k^{p(j)}, & \, \forall (i,k), (j,k) \in I \\
    &&& g() \leq 0 & \, \forall \\
    &&& x_{\text{lb}} \leq x_k^j \leq x_{\text{ub}} & \, \forall \\
    &&& u_{\text{lb}} \leq u_k^j \leq u_{\text{ub}} & \, \forall \\
    &&& z_{\text{lb}} \leq z_k^j \leq z_{\text{ub}} & \, \forall \\
    &&& g_{\text{terminal}}(x_N^j,z_N^j) \leq 0     & \, \forall

where :math:`\tilde{J} = \left(\sum_{i=1}^{N}(\omega_i J_i)^{\alpha}\right)^{1/\alpha}` is the objective.
The objective consists of one part for each scenario, which can be weighted according to the probability of the scenarios :math:`\omega_i`, :math:`i=1,\dots,N_{\text{s}}`.
The cost for each scenario :math:`S_i` is given by:

.. math::

    J_i = m(x_N^j,z_N^n,p_N^o,p_{\text{tv},N})  + \sum_{k=0}^{N-1} l(x_k^q,u_k^r,z_k^s,p_k^t,p_{\text{tv},k}), \quad \forall x_k^{\cdot}, u_k^{\cdot}, z_k^{\cdot}, p_k^{\cdot}, p_{\text{tv},k}^{\cdot} \in S_i,

.. note::

    For all scenarios, which are directly considered in the problem formulation, constraint satisfaction can be guaranteed.
    This means if all uncertainties can only take discrete values, which a represented in the scenario tree, constraint satisfaction can be guaranteed.
    For linear systems, considering the extreme values of the uncertainties in the scenario tree guarantees constraint satisfaction, even if the uncertainties are continuous.

    For nonlinear problems, constraint satisfaction cannot be guaranteed for all possible combinations of the parameters within the uncertainty interval, if the uncertainties are continuous.
    However, often the worst-case scenarios are at the boundaries of the intervals of the uncertain parameters.
    So, if the boundaries are considered in the scenario tree, constraint satisfaction for all cases is highly probable.

.. note::

    It the uncertainties :math:`p_k` are unknown but constant, choosing the robust horizon :math:`N_{\text{robust}}=1` is sufficient to include all reasonable scenarios.
