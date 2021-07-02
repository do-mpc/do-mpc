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
sys.path.append('../../')
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Simple oscillating masses example with two masses and two inputs.
    # States are the position and velocitiy of the two masses.

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    model.set_expression(expr_name='cost', expr=sum1(_x**2))


    A = np.array([[ 0.763,  0.460,  0.115,  0.020],
                  [-0.899,  0.763,  0.420,  0.115],
                  [ 0.115,  0.020,  0.763,  0.460],
                  [ 0.420,  0.115, -0.899,  0.763]])

    B = np.array([[0.014],
                  [0.063],
                  [0.221],
                  [0.367]])


    x_next = A@_x+B@_u
    model.set_rhs('x', x_next)

    model.setup()

    return model
