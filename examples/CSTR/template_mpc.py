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


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
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

    mpc.set_param(**setup_mpc)

    mpc.set_param(store_full_solution=True)

    mpc._x_scaling['T_R'] = 100
    mpc._x_scaling['T_K'] = 100
    mpc._u_scaling['Q_dot'] = 2000
    mpc._u_scaling['F'] = 100

    _x, _u, _z, _tvp, p, _aux,  *_ = mpc.model.get_variables()

    mterm = (_x['C_b'] - 0.6)**2
    lterm = (_x['C_b'] - 0.6)**2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(F=0.1, Q_dot = 1e-3)

    mpc.bounds['lower', '_x', 'C_a'] = 0.1
    mpc.bounds['lower', '_x', 'C_b'] = 0.1
    mpc.bounds['lower', '_x', 'T_R'] = 50
    mpc.bounds['lower', '_x', 'T_K'] = 50

    mpc.bounds['upper', '_x', 'C_a'] = 2
    mpc.bounds['upper', '_x', 'C_b'] = 2
    mpc.bounds['upper', '_x', 'T_K'] = 140

    mpc.bounds['lower', '_u', 'F'] = 5
    mpc.bounds['lower', '_u', 'Q_dot'] = -8500

    mpc.bounds['upper', '_u', 'F'] = 100
    mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

    # Instead of having a regular bound on T_R:
    #mpc.bounds['upper', '_x', 'T_R'] = 140
    # We can also have soft consraints as part of the set_nl_cons method:
    mpc.set_nl_cons('T_R', _x['T_R'], ub=140, soft_constraint=True, penalty_term_cons=1e2)


    mpc._x0['C_a'] = 0.8
    mpc._x0['C_b'] = 0.5
    mpc._x0['T_R'] = 134.14
    mpc._x0['T_K'] = 130.0

    alpha_var = np.array([1., 1.05, 0.95])
    beta_var = np.array([1., 1.1, 0.9])

    mpc.set_uncertainty_values([alpha_var, beta_var])

    mpc.setup()

    return mpc
