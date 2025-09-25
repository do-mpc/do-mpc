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

# imports
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

    # Set settings of MPC:
    setup_mpc = {
        'n_horizon': 25,
        'n_robust': 0,
        'open_loop': 0,
        't_step': .3,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }
    mpc.set_param(**setup_mpc)

    # setting up the cost function
    x_0 = model.x['x_0']
    x_1 = model.x['x_1']
    mterm = (x_0 - 1) ** 2 + (x_1 - 1) ** 2
    lterm = mterm
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Optional, setting up input penalization
    mpc.set_rterm(inp=1)

    # Bounds
    mpc.bounds['lower', '_x', 'x_0'] = 0.0
    mpc.bounds['lower', '_x', 'x_1'] = 0.0
    mpc.bounds['upper', '_x', 'x_0'] = 2.0
    mpc.bounds['upper', '_x', 'x_0'] = 2.0
    mpc.bounds['lower', '_u', 'inp'] = 0.0
    mpc.bounds['upper', '_u', 'inp'] = 1.0

    # completing the setup of the mpc
    mpc.setup()

    # end of function
    return mpc


