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


def template_model():
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(2,1))

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    model.set_expression(expr_name='cost', expr=sum1(_x**2))


    A = np.array([[0.96033209,  0.19734663,  0.01973449,  0.0013227],
                  [-0.39337056,  0.96033209,  0.19470123,  0.01973449],
                  [0.01973449,  0.0013227,  0.96033209,  0.19734663],
                  [0.19470123,  0.01973449, -0.39337056,  0.96033209]])

    B = np.array([[1.98671102e-02, 6.63119354e-05],
                  [1.97346631e-01, 1.32269963e-03],
                  [6.63119354e-05, 1.98671102e-02],
                  [1.32269963e-03, 1.97346631e-01]])


    x_next = A@_x+B@_u
    model.set_rhs('x', x_next)

    model.setup_model()

    return model
