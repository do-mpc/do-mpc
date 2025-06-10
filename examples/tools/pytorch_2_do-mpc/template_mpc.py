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

    mpc = do_mpc.controller.MPC(model=model)

    # suppress solver output
    if silence_solver:
        mpc.settings.supress_ipopt_output()

    mpc.set_param(t_step=0.1)
    mpc.set_param(n_horizon=10)

    setpoint = 0.005
    mterm = (setpoint - model.x['states'][0])**2
    mpc.set_objective(mterm=mterm, lterm=mterm)

    mpc.set_rterm(inputs=0.1)

    lbx = np.array([-0.01, -2.65/100])
    ubx = np.array([0.01, 2.65/100])
    lbu = np.array([-0.1])
    ubu = np.array([0.1])

    mpc.bounds['lower', '_x', 'states'] = lbx
    mpc.bounds['upper', '_x', 'states'] = ubx
    mpc.bounds['lower', '_u', 'inputs'] = lbu
    mpc.bounds['upper', '_u', 'inputs'] = ubu

    mpc.setup()

    return mpc