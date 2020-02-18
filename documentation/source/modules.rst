API
===

Note the following important inheritance of **do mpc** classes:

.. graphviz::
    :name: class_inheritance
    :caption: Class inheritance. Click on classes for more information.
    :align: center

    digraph G {
        graph [fontname = "Consolas"];
        node [fontname = "Consolas", shape=box, fontcolor="#404040", color="#afafaf"];
        edge [fontname = "Consolas"];

        optimizer [label="do_mpc.optimizer.Optimizer", href="../do_mpc.html#module-do_mpc.optimizer", target="_top"];
        MPC [label="do_mpc.controller.MPC", href="../do_mpc.html#do_mpc.controller.MPC", target="_top"];
        MHE [label="do_mpc.estimator.MHE", href="../do_mpc.html#do_mpc.estimator.MHE", target="_top"];
        Estimator [label="do_mpc.estimator.Estimator", href="../do_mpc.html#do_mpc.estimator.Estimator", target="_top"];
        EKF [label="do_mpc.estimator.EKF", href="../do_mpc.html#do_mpc.estimator.EKF", target="_top"];
        StateFeedback [label="do_mpc.estimator.StateFeedback", href="../do_mpc.html#do_mpc.estimator.StateFeedback", target="_top"];

        optimizer -> MPC, MHE;
        Estimator -> EKF, StateFeedback, MHE;
    }

.. toctree::
   :maxdepth: 5

   do_mpc
