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

def template_lqr(model):
    """
    --------------------------------------------------------------------------
    template_lqr: tuning parameters
    --------------------------------------------------------------------------
    """
    t_sample = 0.5
    model_dc = model.discretize(t_sample)
    
    lqr = do_mpc.controller.LQR(model_dc)
    
    # Initialize parameters
    setup_lqr = {
        'n_horizon':10
        }
    lqr.set_param(**setup_lqr)
    
    # Setting objective
    Q = 10*np.identity(5)
    R = 5*np.identity(1)
    # Rdelu=5*np.identity(1) # Weight matrix for rated input
    # delZ = 10*np.identity(1) # Weight matrix for algebraic states
    lqr.set_objective(Q = Q, R = R)
    
    # Setup lqr
    lqr.setup()
    
    return lqr