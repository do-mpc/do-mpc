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


def template_mpc(model, silence_solver=False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)


    # Set settings of MPC:
    mpc.settings.n_horizon = 20
    mpc.settings.n_robust = 1
    mpc.settings.open_loop = 0
    mpc.settings.t_step = 0.005
    mpc.settings.state_discretization = 'collocation'
    mpc.settings.collocation_type = 'radau'
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 1
    mpc.settings.store_full_solution = True

    # suppress solver output
    if silence_solver:
        mpc.settings.supress_ipopt_output()

    # setting up the scaling of the variables
    mpc.scaling['_x', 'T_R'] = 100
    mpc.scaling['_x', 'T_K'] = 100
    mpc.scaling['_u', 'Q_dot'] = 2000
    mpc.scaling['_u', 'F'] = 100

    # setting up the cost function
    mterm = (model.x['C_b'] - 0.6)**2
    lterm = (model.x['C_b'] - 0.6)**2
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # setting up the factors for input penalisation
    mpc.set_rterm(F=0.1, Q_dot = 1e-3)
    
    # setting up lower boundaries for the states
    mpc.bounds['lower', '_x', 'C_a'] = 0.1
    mpc.bounds['lower', '_x', 'C_b'] = 0.1
    mpc.bounds['lower', '_x', 'T_R'] = 50
    mpc.bounds['lower', '_x', 'T_K'] = 50

    # setting up upper boundaries for the states
    mpc.bounds['upper', '_x', 'C_a'] = 2
    mpc.bounds['upper', '_x', 'C_b'] = 2
    mpc.bounds['upper', '_x', 'T_K'] = 140

    # setting up lower boundaries for the inputs
    mpc.bounds['lower', '_u', 'F'] = 5
    mpc.bounds['lower', '_u', 'Q_dot'] = -8500

    # setting up upper boundaries for the inputs
    mpc.bounds['upper', '_u', 'F'] = 100
    mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

    # Instead of having a regular bound on T_R:
    #mpc.bounds['upper', '_x', 'T_R'] = 140
    # We can also have soft constraints as part of the set_nl_cons method:
    mpc.set_nl_cons('T_R', model.x['T_R'], ub=140, soft_constraint=True, penalty_term_cons=1e2)

    # setting up parameter uncertainty
    alpha_var = np.array([1., 1.05, 0.95])
    beta_var = np.array([1., 1.1, 0.9])
    mpc.set_uncertainty_values(alpha = alpha_var, beta = beta_var)

    # completing the setup of the mpc
    mpc.setup()

    # end of function
    return mpc
