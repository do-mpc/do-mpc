Installation
============
**do-mpc** is a python 3.x package. Follow this guide to install **do-mpc**.

If you are new to Python, please read this `article <https://protostar.space/why-you-need-python-environments-and-how-to-manage-them-with-conda>`_
about Python environments. We recommend using a new Python environment for every project and to manage it with miniconda.

Requirements
**************
**do-mpc** requires the following Python packages and their dependencies:

* numpy

* CasADi

* matplotlib


Option 1: **PIP**
*****************
Simply use **PIP** and install **do-mpc** from the terminal.
This has the advantage that **do-mpc** is always in your Python path
and can be used throughout your projects.

1. Install **do-mpc**:

::

    pip install do-mpc

Tested on Windows and Linux (Ubuntu 19.04).

**PIP** will also
take care of dependencies and you are immediately ready to go.

Use this option if you plan to use **do-mpc** without altering the source code,
e.g. write extensions.

2. Get example documents:

All resources can be obtained from our  `release notes`_ page.
Please find the example files that match your currently installed **do-mpc** version
in the respective section.

.. _`release notes`: release_notes.html

Option 2: **Clone from Github**
*******************************
More experienced users are advised to clone or fork the most recent version of **do-mpc**
from `GitHub <https://github.com/do-mpc/do-mpc>`_:

::

    git clone https://github.com/do-mpc/do-mpc.git

In this case, the dependencies from above must be manually taken care of.
You have immediate access to our examples.


HSL linear solver for IPOPT
***************************

The standard configuration of **do-mpc** is based on IPOPT_
to solve the nonlinear constrained optimization problems that arise with the MPC and MHE formulation.
The computational bottleneck of this method is repeatedly solving a large-scale linear systems for which
IPOPT is offering a an interface to a variety of sparse symmetric indefinite linear solver.
IPOPT and thus **do-mpc** comes by default with the MUMPS_ solver.
It is suggested to try a different linear solver for IPOPT with **do-mpc**.
Typically, a significant speed boost can be achieved with the HSL_ MA27 solver.


Option 1: **Pre-compiled binaries**
-----------------------------------

When installing CasADi via PIP or Anaconda
(happens automatically when installing **do-mpc** via PIP),
you obtain the pre-compiled CasADi package.
To use MA27 (or other HSL solver in this setup) please follow these steps:

Linux
^^^^^
(Tested on Ubuntu 19.10)

1. Obtain the HSL_ shared library. Choose the personal licence.

2. Unpack the archive and copy its content to a destination of your choice. (e.g. ``/home/username/Documents/coinhsl/``)

3. Rename ``libcoinhsl.so`` to ``libhsl.so``. CasADi is searching for the shared libraries under a depreciated name.

4. Locate your ``.bashrc`` file on your home directory (e.g. ``/home/username/.bashrc``)

5. Add the previously created directory to your ``LD_LIBRARY_PATH``, by adding the following line to your ``.bashrc``

.. code-block:: console

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/ffiedler/Documents/coinhsl/lib"

6. Install ``libgfortran`` with Anaconda:

.. code-block:: console

    conda install -c anaconda libgfortran


.. note::

    To check if MA27 can be used as intended, please first change the solver according to :py:func:`do_mpc.controller.MPC.set_param`.
    When running the examples, inspect the IPOPT output in the console. Two possible errors are expected:

    .. code-block:: console

        Tried to obtain MA27 from shared library "libhsl.so", but the following error occured:
        libhsl.so: cannot open shared object file: No such file or directory

    This error suggests that step three above wasn't executed or didn't work.

    .. code-block:: console

        Tried to obtain MA27 from shared library "libhsl.so", but the following error occured:
        libgfortran.so.3: cannot open shared object file: No such file or directory

    This error suggests that step six wasn't executed or didn't work.



Option 2: **Compile from source**
---------------------------------------------

Please see the comprehensive guide on the CasADi_ Github Wiki.





.. _CasADi: https://github.com/casadi/casadi/wiki/Obtaining-HSL
.. _IPOPT: https://coin-or.github.io/Ipopt/
.. _MUMPS: http://mumps.enseeiht.fr/
.. _HSL: http://www.hsl.rl.ac.uk/ipopt/
