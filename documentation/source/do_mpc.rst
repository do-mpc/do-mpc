API Reference
=============
Find below a table of all **do-mpc** modules.
Classes and functions of each module are shown on their respective page.

Note the following important inheritance of **do-mpc** classes:

.. graphviz::
    :name: class_inheritance
    :caption: Class inheritance. Click on classes for more information.
    :align: center

    digraph G {
        graph [fontname = "Consolas"];
        node [fontname = "Consolas", shape=box, fontcolor="#404040", color="#707070"];
        edge [fontname = "Consolas", color="#707070"];

        optimizer [label="optimizer.Optimizer", href="../api/do_mpc.optimizer.Optimizer.html#optimizer", target="_top"];
        MPC [label="controller.MPC", href="../api/do_mpc.controller.MPC.html#mpc", target="_top"];
        MHE [label="estimator.MHE", href="../api/do_mpc.estimator.MHE.html#mhe", target="_top"];
        Estimator [label="estimator.Estimator", href="../api/do_mpc.estimator.Estimator.html#do_mpc.estimator.Estimator", target="_top"];
        EKF [label="estimator.EKF", href="../api/do_mpc.estimator.EKF.html#ekf", target="_top"];
        StateFeedback [label="estimator.StateFeedback", href="../api/do_mpc.estimator.StateFeedback.html#statefeedback", target="_top"];

        optimizer -> MPC, MHE;
        Estimator -> EKF, StateFeedback, MHE;
    }

.. automodule:: do_mpc
.. currentmodule:: do_mpc


.. rubric:: Modules
.. autosummary::
    :toctree: api

    model

    simulator

    optimizer

    controller

    estimator

    data

    graphics
