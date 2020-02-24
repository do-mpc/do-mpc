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

Download our examples here_. You can also get the MPC_ Jupyter Notebook and the MHE_ Jupyter Notebook from the introduction.

.. _here: https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/do-mpc/do-mpc/tree/master/examples
.. _MPC: https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/do-mpc/do-mpc/blob/master/documentation/source/getting_started.ipynb
.. _MHE: https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/do-mpc/do-mpc/blob/master/documentation/source/mhe_example.ipynb

All these resources can also be obtained by simply cloning the repository as shown below.

Option 2: **Clone from Github**
*******************************
More experienced users are advised to clone or fork the most recent version of **do-mpc**
from `GitHub <https://github.com/do-mpc/do-mpc>`_:

::

    git clone https://github.com/do-mpc/do-mpc.git

In this case, the dependencies from above must be manually taken care of.
You have immediate access to our examples.
