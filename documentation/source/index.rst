.. do-mpc documentation master file, created by
   sphinx-quickstart on Fri Nov 29 17:20:10 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description:
      do-mpc is a comprehensive open-source Python toolbox for robust model predictive control (MPC)
      and moving horizon estimation (MHE).
   :google-site-verification:
      vPyUwidEDFuOS3MRTnjaah0fEs91dYSHomsAj1mxssA
   :google-site-verification:
      IheinIFIQweRVW75acwPYtDY_KjIGGOUY0VECK9mz2k
   :canonical:
      https://www.do-mpc.com/en/latest/


Model predictive control python toolbox
=======================================

.. image:: https://readthedocs.org/projects/do-mpc/badge/?version=latest
    :target: https://www.do-mpc.com
    :alt: Documentation Status
.. image:: https://travis-ci.org/do-mpc/do-mpc.svg?branch=master
    :target: https://travis-ci.org/do-mpc/do-mpc
    :alt: Build Status
.. image:: https://badge.fury.io/py/do-mpc.svg
    :target: https://badge.fury.io/py/do-mpc

**do-mpc** is a comprehensive open-source toolbox for robust **model predictive control (MPC)**
and **moving horizon estimation (MHE)**.
**do-mpc** enables the efficient formulation and solution of control and estimation problems for nonlinear systems,
including tools to deal with uncertainty and time discretization.
The modular structure of **do-mpc** contains simulation, estimation and control components
that can be easily extended and combined to fit many different applications.

In summary, **do-mpc** offers the following features:

* nonlinear and economic model predictive control

* robust multi-stage model predictive control

* moving horizon state and parameter estimation

* modular design that can be easily extended
​
The **do-mpc** software is Python based and works therefore on any OS with a Python 3.x distribution.
**do-mpc** has been developed by Sergio Lucia and Alexandru Tatulea at the DYN chair of the TU Dortmund lead by Sebastian Engell.
The development is continued at the IOT chair of the TU Berlin by Felix Fiedler and Sergio Lucia.


Example: Robust Multi-stage MPC
*******************************
We showcase an example, where the control task is to regulate the rotating triple-mass-spring system as shown below:

.. image:: anim_disc_3d_uncontrolled.gif

Once excited, the uncontrolled system takes a long time to come to a rest.
To influence the system, two stepper motors are connected to the outermost discs via springs.
The designed controller will result in something like this:

.. image:: anim_disc_3d_ctrl_motor.gif

Assume, we have modeled the system from first principles and identified the parameters in an experiment.
We are especially unsure about the exact value of the inertia of the masses.
With Multi-stage MPC, we can define different scenarios e.g. :math:`\pm 10\%` for each mass and predict as well as optimize multiple state and input trajectories.
This family of trajectories will always obey to set constraints for states and inputs and can be visualized as shown below:

.. image:: anim.gif


Next steps
**********
We suggest you start by skimming over the selected examples below to get an first impression of the above mentioned features.
A great further read for interested viewers is the `getting started: MPC`_ page, where we show how to setup **do-mpc** for the
robust control task of a triple-mass-spring system.
A state and parameter moving horizon estimator is configured and used for the same system in `getting started: MHE`_.

.. _`getting started: MPC`: getting_started.ipynb
.. _`getting started: MHE`: mhe_example.ipynb

To install **do-mpc** please see our `installation instructions`_.

.. _`installation instructions`: installation.html


Table of contents
=================



.. toctree::
   :maxdepth: 5
   :caption: Introduction

   getting_started
   mhe_example

.. toctree::
   :maxdepth: 5
   :caption: Background

   theory_orthogonal_collocation
   theory_mpc
   theory_mhe


.. toctree::
   :maxdepth: 5
   :caption: How to get it?

   license
   installation
   credit

.. toctree::
   :maxdepth: 5
   :caption: How to use it?

   project_structure
   Debugging
   do_mpc
   release_notes

.. toctree::
   :maxdepth: 5
   :caption: Example gallery

   batch_reactor
   CSTR




Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
