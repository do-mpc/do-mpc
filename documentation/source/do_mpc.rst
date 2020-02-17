do mpc toolbox
==============

Note the following important inheritance of **do mpc** classes:

.. graphviz::
    :name: class_inheritance
    :caption: Class inheritance.
    :align: center

    digraph G {
        graph [fontname = "Monaco"];
        node [fontname = "Monaco", shape=box];
        edge [fontname = "Monaco"];

        "do_mpc.optimizer.Optimizer" -> "do_mpc.controller.MPC", "do_mpc.estimator.MHE"
        "do_mpc.estimator.Estimator" -> "do_mpc.estimator.EKF", "do_mpc.estimator.StateFeedback", "do_mpc.estimator.MHE"
    }

do\_mpc.model
--------------------

.. automodule:: do_mpc.model
   :members:
   :undoc-members:
   :show-inheritance:

do\_mpc.optimizer
------------------------

.. automodule:: do_mpc.optimizer
   :exclude-members: prepare_data
   :members:
   :undoc-members:
   :show-inheritance:

do\_mpc.controller
------------------------

.. automodule:: do_mpc.controller
   :exclude-members: _check_validity
   :members:
   :undoc-members:
   :show-inheritance:

do\_mpc.estimator
-------------------

.. automodule:: do_mpc.estimator
   :exclude-members: check_validity
   :members:
   :undoc-members:
   :show-inheritance:

do\_mpc.simulator
------------------------

.. automodule:: do_mpc.simulator
   :members:
   :undoc-members:
   :show-inheritance:

do\_mpc.data
-------------------

.. automodule:: do_mpc.data
   :members:
   :undoc-members:
   :show-inheritance:
