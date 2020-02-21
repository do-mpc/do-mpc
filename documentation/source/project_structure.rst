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

        Model [label="Model", href="../do_mpc.html#do_mpc.model.Model", target="_top", shape=box, style=filled]
        MPC [href="../do_mpc.html#do_mpc.controller.MPC", target="_top", shape=box, style=filled]
        Simulator [href="../do_mpc.html#do_mpc.simulator.Simulator", target="_top", shape=box, style=filled]
        MHE [href="../do_mpc.html#do_mpc.estimator.MHE", target="_top", shape=box, style=filled]

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

As shown above, we split our MHE / MPC configuration into four seperate files:

* ``template_model``: Define the dynamic ``Model`` (states, inputs, dynamic, etc.)

* ``template_mpc``: Configure the MPC controller based on the ``Model`` (objective, bounds, etc.)

* ``template_simulator``: Configure the DAE/ODE/discrete simulator based on the ``Model`` (objective, bounds, etc.)

* ``template_estimator``: Configure the estimator (MHE / EKF / state-feedback) based on the ``Model``.

These python scripts usually consist of a single function, e.g. for the optimizer:

::

    def optimizer(model):
        mpc = do_mpc.controller.MPC(model)

        % configuration.

        return mpc

and return the :py:mod:`do_mpc.model.Model`, :py:mod:`do_mpc.controller.MPC`, :py:mod:`do_mpc.simulator.Simulator`
or :py:mod:`do_mpc.estimator.MHE` objects.

All these functions are called from a single ``main.py`` file, e.g.:

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


Main loop
#########
As shown in Diagram :ref:`project_structure`, after obtaining the different **do mpc**
objects they can be used in the *main loop*. In code form the loop looks like this:

::

    for k in range(N_iterations):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
