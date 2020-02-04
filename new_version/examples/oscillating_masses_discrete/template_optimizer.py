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


def template_optimizer(model):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    optimizer = do_mpc.optimizer(model)

    setup_optimizer = {
        'n_robust': 0,
        'n_horizon': 20,
        't_step': 0.1,
        'state_discretization': 'discrete',
    }

    optimizer.set_param(**setup_optimizer)

    _x, _u, _z, _tvp, p, _aux, *_ = optimizer.model.get_variables()

    mterm = _aux['x_squared']
    lterm = sum1(_x.cat**2)

    optimizer.set_objective(mterm=mterm, lterm=lterm)
    rterm_factor = optimizer.get_rterm()
    rterm_factor['u1'] = 1e-4
    rterm_factor['u2'] = 1e-4

    optimizer._x_lb['x'] = -3
    optimizer._x_ub['x'] = 3

    optimizer._u_lb['u1'] = -5
    optimizer._u_ub['u1'] = 5
    optimizer._u_lb['u2'] = -5
    optimizer._u_ub['u2'] = 5

    optimizer._x0['x'] = 0.5

    optimizer.set_nl_cons(x_max=_x['x'])
    optimizer._nl_cons_ub['x_max'] = 100

    optimizer.set_uncertainty_values(np.array([[1.0,1.2],[2.0,2.3]]))
    optimizer.setup()

    return optimizer
