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
        'n_horizon': 20,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 1.0,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
    }

    optimizer.set_param(**setup_optimizer)

    _x, _u, _z, _tvp, p, _aux = optimizer.model.get_variables()

    mterm = -_x['P_s']
    lterm = -_x['P_s']

    optimizer.set_objective(mterm=mterm, lterm=lterm)
    rterm_factor = optimizer.get_rterm()
    rterm_factor['inp'] = 1.0

    optimizer._x_lb['X_s'] = 0.0
    optimizer._x_lb['S_s'] = -0.01
    optimizer._x_lb['P_s'] = 0.0
    optimizer._x_lb['V_s'] = 0.0

    optimizer._x_ub['X_s'] = 3.7
    optimizer._x_ub['S_s'] = inf
    optimizer._x_ub['P_s'] = 3.0
    optimizer._x_ub['V_s'] = inf

    optimizer._u_lb['inp'] = 0.0

    optimizer._u_ub['inp'] = 0.2

    optimizer._x0['X_s'] = 1.0
    optimizer._x0['S_s'] = 0.5
    optimizer._x0['P_s'] = 0.0
    optimizer._x0['V_s'] = 120.0

    optimizer.set_nl_cons(x_max=_u['inp'])
    optimizer._nl_cons_ub['x_max'] = 100000

    Y_x_values = np.array([0.5, 0.4, 0.3])
    S_in_values = np.array([200.0, 220.0, 180.0])

    optimizer.set_uncertainty_values([Y_x_values, S_in_values])

    optimizer.setup()

    return optimizer
