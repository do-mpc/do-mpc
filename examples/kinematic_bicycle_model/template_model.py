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

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os

rel_do_mpc_path = os.path.join('..', '..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """

    "Chronos and CRS: Design of a miniature car-like robot and a software framework for single and multi-agent robotics and control"
    model_type = 'continuous'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters
    m = 2.0  # Mass [kg]
    lf = 0.3  # Distance from CoG to front wheel [m]
    lr = 0.3  # Distance from CoG to rear wheel [m]
    w = 0.15  # Width of the car [m]

    # States struct (optimization variables):
    X_p = model.set_variable(var_type='_x', var_name='X_p', shape=(1, 1))
    Y_p = model.set_variable(var_type='_x', var_name='Y_p', shape=(1, 1))
    Psi = model.set_variable(var_type='_x', var_name='Psi', shape=(1, 1))
    V = model.set_variable(var_type='_x', var_name='V', shape=(1, 1))

    # Input struct (optimization variables):
    Delta = model.set_variable(var_type='_u', var_name='Delta')
    Acc = model.set_variable(var_type='_u', var_name='Acc')

    # # Set expression. These can be used in the cost function, as non-linear constraints
    # # or just to monitor another output.
    # Expressions can also be formed without beeing explicitly added to the model.
    # The main difference is that they will not be monitored and can only be used within the current file.
    Beta = atan((lr / (lr + lf)) * tan(Delta))
    # Differential equations
    model.set_rhs('X_p', V * cos(Psi + Beta))
    model.set_rhs('Y_p', V * sin(Psi + Beta))
    model.set_rhs('Psi', (V / lr) * sin(Beta))
    model.set_rhs('V', Acc)

    # Build the model
    model.setup()

    # end of function
    return model
