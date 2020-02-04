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
        'n_robust': 1,
        'open_loop': 0,
        't_step': 0.005,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 1,
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    optimizer.set_param(**setup_optimizer)

    optimizer.set_param(store_full_solution=True)

    optimizer._x_scaling['T_R'] = 100
    optimizer._x_scaling['T_K'] = 100
    optimizer._u_scaling['Q_dot'] = 2000
    optimizer._u_scaling['F'] = 100

    _x, _u, _z, _tvp, p, _aux = optimizer.model.get_variables()

    mterm = (_x['C_b'] - 0.6)**2
    lterm = (_x['C_b'] - 0.6)**2

    optimizer.set_objective(mterm=mterm, lterm=lterm)

    optimizer.set_rterm(F=0.1, Q_dot = 1e-3)

    optimizer.bounds['lower', '_x', 'C_a'] = 0.1
    optimizer.bounds['lower', '_x', 'C_b'] = 0.1
    optimizer.bounds['lower', '_x', 'T_R'] = 50
    optimizer.bounds['lower', '_x', 'T_K'] = 50

    optimizer.bounds['upper', '_x', 'C_a'] = 2
    optimizer.bounds['upper', '_x', 'C_b'] = 2
    optimizer.bounds['upper', '_x', 'T_R'] = 140
    optimizer.bounds['upper', '_x', 'T_K'] = 140

    optimizer.bounds['lower', '_u', 'F'] = 5
    optimizer.bounds['lower', '_u', 'Q_dot'] = -8500

    optimizer.bounds['upper', '_u', 'F'] = 100
    optimizer.bounds['upper', '_u', 'Q_dot'] = 0.0


    optimizer._x0['C_a'] = 0.8
    optimizer._x0['C_b'] = 0.5
    optimizer._x0['T_R'] = 134.14
    optimizer._x0['T_K'] = 130.0

    alpha_var = np.array([1., 1.05, 0.95])
    beta_var = np.array([1., 1.1, 0.9])

    optimizer.set_uncertainty_values([alpha_var, beta_var])

    optimizer.setup()

    return optimizer
