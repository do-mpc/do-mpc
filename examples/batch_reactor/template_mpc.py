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
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_mpc(model, silence_solver = False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    # Set settings of MPC:
    mpc.settings.n_horizon =  20
    mpc.settings.n_robust =  0
    mpc.settings.open_loop =  0
    mpc.settings.t_step =  1.0
    mpc.settings.state_discretization =  'collocation'
    mpc.settings.collocation_type =  'radau'
    mpc.settings.collocation_deg =  2
    mpc.settings.collocation_ni =  2
    mpc.settings.store_full_solution =  True

    # suppress solver output
    if silence_solver:
        mpc.settings.supress_ipopt_output()

    # setting up the cost function
    mterm = -model.x['P_s']
    lterm = -model.x['P_s']
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # setting up the factor for the inputs
    mpc.set_rterm(inp=1.0)

    # setting up lower boundaries for the states
    mpc.bounds['lower', '_x', 'X_s'] = 0.0
    mpc.bounds['lower', '_x', 'S_s'] = -0.01
    mpc.bounds['lower', '_x', 'P_s'] = 0.0
    mpc.bounds['lower', '_x', 'V_s'] = 0.0

    # setting up upper boundaries for the states
    mpc.bounds['upper', '_x','X_s'] = 3.7
    mpc.bounds['upper', '_x','P_s'] = 3.0

    # setting up boundaries for the inputs
    mpc.bounds['lower','_u','inp'] = 0.0
    mpc.bounds['upper','_u','inp'] = 0.2

    # setting up parameter uncertainty
    Y_x_values = np.array([0.5, 0.4, 0.3])
    S_in_values = np.array([200.0, 220.0, 180.0])
    mpc.set_uncertainty_values(Y_x = Y_x_values, S_in = S_in_values)

    # completing the setup of the mpc
    mpc.setup()

    # end of function
    return mpc
