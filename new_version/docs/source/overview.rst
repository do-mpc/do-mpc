Overview
========

Welcome to "**do mpc**", an comprehensive toolbox for **robust Model Predictive Control**.
Among others, "**do mpc**" has the following features:

* robust multi-stage MPC

* non-linear and economic MPC (NMPC / EMPC)

* DAE support

* A modular design, completed with state-of-the-art DAE simulator and state-estimation.

* Moving Horizon Estimation (MHE), as well as Extended Kalman Filter for state-estimation.

* An intuitive interface

* Low-level interface and integration with Matplotlib for fully customizable, publication ready graphics and animations.

* comprehensible python code for easy customizability

Robust Multi-stage MPC
**********************
We showcase an example, where the control task is to regulate the rotating triple-mass-spring system as shown below:

.. image:: anim_disc_3d_uncontrolled.gif
Once excited, the uncontrolled system takes a long time to come to a rest. 
To incluence the system, two steper motors are connected to the outermost discs via springs.
The designed controller will result in something like this:

.. image:: anim_discs_3d.gif
Assume, we have modeled the system from first principles and identified the parameters in an experiment.
We are especially unsure about the exact value of the inertia of the masses. 
With Multi-stage MPC, we can define different scenarios e.g. :math:`\pm 10\%` for each mass and predict as well as optimize multiple state and input trajectories.
This family of trajectories will always obey to set constraints for states and inputs and can be visualized as shown below:

.. image:: anim.gif

