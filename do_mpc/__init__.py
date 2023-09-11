#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

"""
Find below a table of all **do-mpc** modules.
Classes and functions of each module are shown on their respective page.

The core modules are used to create the **do-mpc** control loop (click on elements to open documentation page):

.. graphviz::
    :name: control_loop
    :caption: **do-mpc** control loop and interconnection of classes.
    :align: center

    digraph G {
        graph [fontname = "Monaco"];
        node [fontname = "Monaco", fontcolor="#404040", color="#bdbdbd"];
        edge [fontname = "Monaco", color="#707070"];

        Model [label="Model", href="../api/do_mpc.model.Model.html#model", target="_top", shape=box, style=filled]
        MPC [href="../api/do_mpc.controller.MPC.html#mpc", target="_top", shape=box, style=filled]
        Simulator [href="../api/do_mpc.simulator.Simulator.html#simulator", target="_top", shape=box, style=filled]
        MHE [href="../api/do_mpc.estimator.MHE.html#mhe", target="_top", shape=box, style=filled]
        Data_MPC [label="MPCData", href="../api/do_mpc.data.MPCData.html#mpcdata", target="_top", shape=box, style=filled]
        Data_Sim [label="Data", href="../api/do_mpc.data.Data.html#data", target="_top", shape=box, style=filled]
        Data_MHE [label="Data", href="../api/do_mpc.data.Data.html#data", target="_top", shape=box, style=filled]
        Graphics [label="Graphics", href="../api/do_mpc.graphics.Graphics.html#graphics", target="_top", shape=box, style=filled]

        Model -> MPC;
        Model -> Simulator;
        Model -> MHE;

        Model [shape=box, style=filled]

        subgraph cluster_loop {{
            rankdir=TB;
            rank=same;
            MPC -> Simulator [label="inputs"];
            Simulator -> MHE [label="meas."];
            MHE -> MPC [label="states"];
        }}

        MPC -> Data_MPC;
        Simulator -> Data_Sim;
        MHE -> Data_MHE;

        Data_MPC -> Graphics;
        Data_Sim -> Graphics;
        Data_MHE -> Graphics;


    }
"""

from packaging import version
import warnings
import importlib

# Check if optional toollboxes were installed (pip install do_mpc[full])
if importlib.util.find_spec("onnx"):
    import onnx
    __ONNX_INSTALLED__ = True
else:
    __ONNX_INSTALLED__ = False

if importlib.util.find_spec("asyncua"):
    import asyncua
    __ASYNCUA_INSTALLED__ = True
else:
    __ASYNCUA_INSTALLED__ = False

from . import tools
from . import model
from . import optimizer
from . import controller
from . import estimator
from . import simulator
from . import graphics
from . import sampling
from . import data
from . import graphics
from . import sysid
from . import differentiator
from . import opcua

from ._version import __version__


import casadi

# From within Sphinx, casadi.__version__ is not available
try:
    casadi_version_check = version.parse(casadi.__version__) < version.parse("3.6.0")
except:
    casadi_version_check = False

if casadi_version_check:
    warnings.warn("It is recommended to use CasADi version 3.6.0 or higher. Future versions of do-mpc might not be compatible with older versions of CasADi.")
    CASADI_LEGACY_MODE = True

else:
    CASADI_LEGACY_MODE = False
