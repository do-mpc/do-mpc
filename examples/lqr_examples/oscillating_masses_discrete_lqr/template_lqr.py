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
import casadi as cas
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

def template_lqr(model):
    """
    --------------------------------------------------------------------------
    template_lqr: tuning parameters
    --------------------------------------------------------------------------
    """
    # Initialize the controller
    lqr = do_mpc.controller.LQR(model)

    # Initialize the parameters
    lqr.settings.t_step = .5
    lqr.settings.n_horizon = None # infinite horizon

    # Setting the objective
    Q = np.identity(4)
    R = np.identity(1)
    Rdelu = np.identity(1)

    lqr.set_objective(Q=Q, R=R)
    lqr.set_rterm(delR = Rdelu)

    # lqr setup
    lqr.setup()

    return lqr