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
        't_step': 0.005,
        'state_discretization': 'collocation',
    }

    optimizer.set_param(**setup_optimizer)

    _x, _u, _z, _tvp, p, _aux = optimizer.model.get_variables()

    mterm = (_x['C_b'] - 1.0)**2
    lterm = (_x['C_b'] - 1.0)**2

    optimizer.set_objective(mterm=mterm, lterm=lterm)
    rterm_factor = optimizer.get_rterm()
    rterm_factor['F'] = 0
    rterm_factor['Q_dot'] = 0

    optimizer._x_lb['C_a'] = 0.1
    optimizer._x_lb['C_b'] = 0.1
    optimizer._x_lb['T_R'] = 50
    optimizer._x_lb['T_K'] = 50

    optimizer._x_ub['C_a'] = 2.0
    optimizer._x_ub['C_b'] = 2.0
    optimizer._x_ub['T_R'] = 180
    optimizer._x_ub['T_K'] = 180

    optimizer._u_lb['F'] = 5
    optimizer._u_lb['Q_dot'] = -8500

    optimizer._u_ub['F'] = 100
    optimizer._u_ub['Q_dot'] = 0.0

    optimizer._x0['C_a'] = 0.8
    optimizer._x0['C_b'] = 0.5
    optimizer._x0['T_R'] = 134.14
    optimizer._x0['T_K'] = 130.0

    optimizer.set_nl_cons(x_max=_u['F'])
    optimizer._nl_cons_ub['x_max'] = 100000

    optimizer.set_uncertainty_values(np.array([[1.0,1.0],[1.0,1.0]]))

    optimizer.setup_nlp()

    return optimizer
