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
    model = do_mpc.model(model_type)

    # States struct (optimization variables):
    _x = struct_symSX([
        entry('x', shape=(4, 1)),
    ])
    model.set_variables(_x=_x)

    # Input struct (optimization variables):
    _u = struct_symSX([
        entry('u', shape=(2, 1)),
    ])
    model.set_variables(_u=_u)

    _z = struct_symSX([
        entry('dummy', shape=(1, 1))
    ])
    model.set_variables(_z=_z)

    # time-varying parameter struct (parameters for optimization problem):
    _tvp = struct_symSX([
        entry('a', shape=(2, 1)),
    ])
    model.set_variables(_tvp=_tvp)

    # Fixed parameters:
    _p = struct_symSX([
        entry('dummy_1', shape=(2, 1)),
    ])
    model.set_variables(_p=_p)

    A = np.array([[0.96033209,  0.19734663,  0.01973449,  0.0013227],
                  [-0.39337056,  0.96033209,  0.19470123,  0.01973449],
                  [0.01973449,  0.0013227,  0.96033209,  0.19734663],
                  [0.19470123,  0.01973449, -0.39337056,  0.96033209]])

    B = np.array([[1.98671102e-02, 6.63119354e-05],
                  [1.97346631e-01, 1.32269963e-03],
                  [6.63119354e-05, 1.98671102e-02],
                  [1.32269963e-03, 1.97346631e-01]])

    rhs = model.get_rhs()
    rhs['x'] = A@_x['x']+B@_u['u']

    model.set_aux(x_squared=sum1(_x.cat**2))

    return model
