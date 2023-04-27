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
        'n_horizon': 10,
        'n_robust': 0,
        'open_loop': False,
        't_step': 1.0,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.fixed_variable_treatment': 'make_constraint','ipopt.tol': 1e-16}
    }

    mpc.set_param(**setup_mpc)

    mterm = -model.x['P_s']
    lterm = -model.x['P_s']

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(inp=1.0)

    # soft constraint
    # mpc.set_nl_cons('soft_constraint', -model.x['S_s'], ub=0.0, soft_constraint=True, penalty_term_cons=1, maximum_violation=np.inf)
    # mpc.set_nl_cons('soft_constraint', -model.u['inp'], ub=0.2, soft_constraint=True, penalty_term_cons=1, maximum_violation=np.inf)

    mpc.bounds['lower', '_x', 'X_s'] = 0.0
    mpc.bounds['lower', '_x', 'S_s'] = -0.01
    mpc.bounds['lower', '_x', 'P_s'] = 0.0
    mpc.bounds['lower', '_x', 'V_s'] = 0.0

    mpc.bounds['upper', '_x','X_s'] = 3.7
    mpc.bounds['upper', '_x','P_s'] = 3.0

    mpc.bounds['lower','_u','inp'] = 0.0
    mpc.bounds['upper','_u','inp'] = 0.2

    Y_x_values = np.array([0.5, 0.4, 0.3])
    S_in_values = np.array([200.0, 220.0, 180.0])
    # Y_x_values = np.array([0.4])
    # S_in_values = np.array([220.0])
    
    mpc.set_uncertainty_values(Y_x = Y_x_values, S_in = S_in_values)

    mpc.setup()

    return mpc
