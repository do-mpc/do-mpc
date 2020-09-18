************************
Structuring your project
************************
In this guide we show you a suggested structure for your MPC or MHE project.

In general, we advice to use the provided templates from our GitHub_ repository
as a starting point. We will explain the structure following the ``CSTR`` example.
Simple projects can also be developed as presented in our introductory Jupyter Notebooks (`MPC`_, `MHE`_)

.. _GitHub: https://github.com/do-mpc/do-mpc
.. _MPC: getting_started.ipynb
.. _MHE: mhe_example.ipynb


.. graphviz::
    :name: project_structure
    :caption: Project structure
    :align: center

    digraph G {
        graph [fontname = "Monaco"];
        node [fontname = "Monaco", fontcolor="#404040", color="#bdbdbd"];
        edge [fontname = "Monaco", color="#707070"];

        Model [label="Model", href="../api/do_mpc.model.Model.html#model", target="_top", shape=box, style=filled]
        MPC [href="../api/do_mpc.controller.MPC.html#mpc", target="_top", shape=box, style=filled]
        Simulator [href="../api/do_mpc.simulator.Simulator.html#simulator", target="_top", shape=box, style=filled]
        MHE [href="../api/do_mpc.estimator.MHE.html#mhe", target="_top", shape=box, style=filled]
        template_model [href="../project_structure.html#template-model", target="_top"];
        template_mpc [href="../project_structure.html#template-mpc", target="_top"];
        template_simulator [href="../project_structure.html#template-simulator", target="_top"];
        template_estimator [href="../project_structure.html#template-estimator", target="_top"];

        template_model -> Model
        Model -> template_mpc, template_simulator, template_estimator;

        Model [shape=box, style=filled]

        subgraph cluster_loop {{
            rankdir=TB;
            rank=same;
            MPC -> Simulator [label="inputs"];
            Simulator -> MHE [label="meas."];
            MHE -> MPC [label="states"];
        }}
        template_mpc -> MPC;
        template_simulator -> Simulator;
        template_estimator -> MHE;
    }

We split our MHE / MPC configuration into five separate files:

========================= ======================================================
``template_model.py``     Define the dynamic model
``template_mpc.py``       Configure the MPC controller
``template_simulator.py`` Configure the DAE/ODE/discrete simulator
``template_estimator.py`` Configure the estimator (MHE / EKF / state-feedback)
``main.py``               **Obtain all configured modules and run the loop.**
========================= ======================================================


The files all include a single function and return the configured :py:mod:`do_mpc.model.Model`,
:py:mod:`do_mpc.controller.MPC`, :py:mod:`do_mpc.simulator.Simulator`
or :py:mod:`do_mpc.estimator.MHE` objects, when called from a central ``main.py`` script.

template_model
**************
The **do-mpc** model class is at the core of all other components and contains the
mathematical description of the investigated dynamical system in the form of
ordinary differential equations (ODE) or differential algebraic equations (DAE).

The ``template_model.py`` file will be structured as follows:

::

    def template_model():
        # Obtain an instance of the do-mpc model class
        # and select time discretization:
        model_type = 'continuous' # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type)

        # Introduce new states, inputs and other variables to the model, e.g.:
        C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
        ...

        Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')
        ...

        # Set right-hand-side of ODE for all introduced states (_x).
        # Names are inherited from the state definition.
        model.set_rhs('C_b', ...)

        # Setup model:
        model.setup()

        return model

template_mpc
************
With the configured model, it is possible to configure and setup the MPC controller.
Note that the optimal control problem (OCP) is always given in the following form:

.. math::

    &\min_{x,u,z}\quad &\sum_{k=0}^{N}\left( \underbrace{l(x_k,u_k,z_k,p)}_{\text{lagrange term}}
    + \underbrace{\Delta u_k^T R \Delta u_k}_{\text{r-term}}\right)
    + &\underbrace{m(x_{N+1})}_{\text{meyer term}}\\
    &\text{subject to:} &\quad x_{\text{lb}} \leq x_k \leq x_{\text{ub}} & \forall k=0,\dots, N+1 \\
    & &\quad u_{\text{lb}} \leq u_k \leq u_{\text{ub}} & \forall k=0,\dots, N\\
    & &\quad z_{\text{lb}} \leq z_k \leq z_{\text{ub}} & \forall k=0,\dots, N\\
    & & m\left(x_k, u_k, z_k, p_k, p_k^{\text{tv}}\right) \leq m_{\text{ub}} & \forall k=0,\dots, N

The configuration of the :py:mod:`do_mpc.controller.MPC` class in ``template_mpc.py`` can be done as follows:

::

    def template_mpc(model):
        # Obtain an instance of the do-mpc MPC class
        # and initiate it with the model:
        mpc = do_mpc.controller.MPC(model)

        # Set parameters:
        setup_mpc = {
            'n_horizon': 20,
            'n_robust': 1,
            't_step': 0.005,
            ...
        }
        mpc.set_param(**setup_mpc)

        # Configure objective function:
        mterm = (_x['C_b'] - 0.6)**2    # Setpoint tracking
        lterm = (_x['C_b'] - 0.6)**2    # Setpoint tracking

        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(F=0.1, Q_dot = 1e-3) # Scaling for quad. cost.

        # State and input bounds:
        mpc.bounds['lower', '_x', 'C_b'] = 0.1
        mpc.bounds['upper', '_x', 'C_b'] = 2.0
        ...

        mpc.setup()

        return mpc


template_simulator
******************
In many cases a developed control approach is first tested on a simulated system.
**do-mpc** responds to this need with the ``simulator`` class.
The ``simulator`` uses state-of-the-art DAE solvers, e.g. Sundials CVODE_ to solve the DAE equations defined in the supplied ``model``.
This will often be the same model as defined for the ``optimizer`` but it is also possible to use a more complex model of the same system.

.. _CVODE: https://computing.llnl.gov/projects/sundials/cvode

The simulator is configured and setup with the supplied ``model`` in the ``template_simulator.py`` file,
which is structured as follows:

::

    def template_simulator(model):
        # Obtain an instance of the do-mpc simulator class
        # and initiate it with the model:
        simulator = do_mpc.simulator.Simulator(model)

        # Set parameter(s):
        simulator.set_param(t_step = 0.005)

        # Optional: Set function for parameters and time-varying parameters.

        # Setup simulator:
        simulator.setup()

        return simulator

template_estimator
******************
In the case that a dedicated estimator is required, another python file should be added to the
project. Configuration and setup of the moving horizon estimator (MHE) will be structured as follows:

::

    def template(mhe):
        # Obtain an instance of the do-mpc MHE class
        # and initiate it with the model.
        # Optionally pass a list of parameters to be estimated.
        mhe = do_mpc.estimator.MHE(model)

        # Set parameters:
        setup_mhe = {
            'n_horizon': 10,
            't_step': 0.1,
            'meas_from_data': True,
        }
        mhe.set_param(**setup_mhe)

        # Set custom objective function
        # based on:
        y_meas = mhe._y_meas
        y_calc = mhe._y_calc

        # and (for the arrival cost):
        x_0 = mhe._x
        x_prev = mhe._x_prev

        ...
        mhe.set_objective(...)

        # Set bounds for states, parameters, etc.
        mhe.bounds[...] = ...

        # [Optional] Set measurement function.
        # Measurements are read from data object by default.

        mhe.setup()

        return mhe

Note that the cost function for the MHE can be freely configured using the available variables.
Generally, we suggest to choose the typical MHE formulation:


.. math::

    J=  &\underbrace{(x_0 - \tilde{x}_0)^T P_x (x_0 - \tilde{x}_0)}_{\text{arrival cost states}} +
        \underbrace{(p_0 - \tilde{p}_0)^T P_p (p_0 - \tilde{p}_0)}_{\text{arrival cost params.}} \\
        &+\sum_{k=0}^{n-1} \underbrace{(h(x_k, u_k, p_k) - y_k)^T P_{y,k} (h(x_k, u_k, p_k) - y_k)}_{\text{stage cost}}

The measurement function must be defined in the model definition and typically contains
the inputs. Inputs are not treated separately as in some other formulations.

main script
***********

All previously defined functions are called from a single ``main.py`` file, e.g.:

::

    from template_model import template_model
    from template_mpc import template_mpc
    from template_simulator import template_simulator

    model = template_model()
    mpc = template_mpc(model)
    simulator = template_simulator(model)
    estimator = do_mpc.estimator.StateFeedback(model)

Simple configurations, as for the :py:mod:`do_mpc.estimator.StateFeedback`
class above are often directly implemented in the ``main.py`` file.

Initial state & guess
#####################

Afterwards we set the initial state (true state) for the simulator.
Note that in proper investigations we usually have a different initial state
for the ``simulator`` (true state) and e.g. the estimator.

::

    # Set the initial state of simulator:
    C_a_0 = 0.8
    ...
    x0 = np.array([C_a_0, ...]).reshape(-1,1)

    simulator.x0 = x0

We can set the initial guessed state for the MHE by modifying its attribute
similarly as for the simulator shown above. The MPC initial guess is given when
calling the function :py:func:`do_mpc.controller.MPC.make_step` for the first time.

Graphics configuration
######################

Visualization the estimation and control results is key to evaluating performance
and identifying potential problems. **do-mpc** has a powerful graphics library based on
Matplotlib for quick and customizable graphics.
After creating a blank class instance and initiating a figure object with:

::

    # Initialize graphic:
    graphics = do_mpc.graphics.Graphics()

    fig, ax = plt.subplots(5, sharex=True)

we need to configure where and what to plot, with the :py:func:`graphics.Graphics.add_line` method:

::

    graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
    # Fully customizable:
    ax[0].set_ylabel('c [mol/l]')
    ax[0].set_ylim(...)
    ...

Note that we are not plotting anything just yet.


closed-loop
###########

As shown in Diagram :ref:`project_structure`, after obtaining the different **do-mpc**
objects they can be used in the *main loop*. In code form the loop looks like this:

::


    for k in range(N_iterations):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

Instead of running for a fixed number of iterations, we can also start an infinite loop with:

::

    while True:
        ...

or have some checks active:

::

    while mpc._x0['C_b'] <= 0.8:
        ...

During or after the loop, we are using the previously configured ``graphics`` class.
Open-loop predictions can be plotted at each sampling time:

::

    for k in range(N_iterations):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

        graphics.reset_axes()
        graphics.plot_results(mpc.data, linewidth=3)
        graphics.plot_predictions(mpc.data, linestyle='--', linewidth=1)
        plt.show()
        input('next step')

Furthermore, we can obtain a visualization of the full closed-loop trajectory after the loop:

::

    graphics.plot_results(mpc.data)
