Installation
============
**do mpc** is a python 3.x package. Follow this guide to install **do mpc**.

If you are new to Python, please read this `article <https://protostar.space/why-you-need-python-environments-and-how-to-manage-them-with-conda>`_
about Python environments. We recommend using a new Python environment for every project and to manage it with miniconda.

Requirements
**************
**do mpc** requires the following Python packages and their dependencies:

* numpy

* CasADi

* matplotlib


Quick and easy
**************
Simply use **PIP** and install **do mpc** from the terminal.

1. Activate your environment, e.g. (on linux):

::

    source activate [myenvironment]

2. Install **do mpc**:

::
    
    pip install do-mpc

.. warning::
    Only the development version is currently available.

**do mpc** is now available within your selected environment.
If the required packages are not available **PIP** will automatically install them.


Custom and extensible
*********************
More experienced users are adviced to clone or fork the most recent version of **do mpc**
from `GitHub <https://github.com/do-mpc/do-mpc>`_.
In this case, the dependencies from above must be manually taken care of.
