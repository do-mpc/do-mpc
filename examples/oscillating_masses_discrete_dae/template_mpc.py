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
    mpc.settings.n_robust = 0
    mpc.settings.n_horizon = 7
    mpc.settings.t_step = 0.5
    mpc.settings.store_full_solution =True

    # suppress solver output
    if silence_solver:
        mpc.settings.supress_ipopt_output()

    # setting up the cost function
    mterm = model.aux['cost']
    lterm = model.aux['cost'] # terminal cost
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # setting up the factors for input penalisation
    mpc.set_rterm(u=1e-4)

    max_x = np.array([[4.0], [10.0], [4.0], [10.0]])

    # setting up boundaries for the states
    mpc.bounds['lower','_x','x'] = -max_x
    mpc.bounds['upper','_x','x'] =  max_x

    # setting up boundaries for the inputs
    mpc.bounds['lower','_u','u'] = -0.5
    mpc.bounds['upper','_u','u'] =  0.5

    # completing the setup of the mpc
    mpc.setup()

    # end of function
    return mpc
