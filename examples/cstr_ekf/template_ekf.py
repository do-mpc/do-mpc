#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2020 Sergio Lucia, Alexandru Tatulea-Codrean
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


def template_ekf(model):
    """
    --------------------------------------------------------------------------
    template_mhe: tuning parameters
    --------------------------------------------------------------------------
    """
    "Create an EKF object. By default a state-only estimation structure is created"
    "If parameter estimation is desired, a subset of available model parameters must be defined"
    ekf = do_mpc.estimator.EKF(model,['alpha']) # Template for parameters: ['alpha',('beta')]
    
    "IMPORTANT Note: Make sure that all matrices and functions you pass to the EKF have correct structures and sizes!"
    #C = np.matrix([[0,0,1,0],[0,0,0,1]])          # Only temperatures are measured, no estimated parameters
    C = np.matrix([[0,0,1,0,0],[0,0,0,1,0]])      # Estimate only 'alpha', one of the reaction rate coefficients
    #C = np.matrix([[0,0,1,0,0,0],[0,0,0,1,0,0]])  # Estimate both 'alpha' and 'beta'
    
    # Basic tuning for state estimation in the CSTR, with high model credibility 
    Q11 = np.matrix([[0.01, 0.0, 0.0, 0.0],[0.0, 0.01, 0.0, 0.0],[0.0, 0.0, 0.01, 0.0],[0.0, 0.0, 0.0, 0.01]])
    # Different tunning for CSTR state estimation, lower model credibility relative to measurements
    Q12 = np.matrix([[0.1, 0.0, 0.0, 0.0],[0.0, 0.1, 0.0, 0.0],[0.0, 0.0, 0.2, 0.0],[0.0, 0.0, 0.0, 0.5]])      
    # New Q matrix for state-AND-parameter estimation in the CSTR (note one extra column and row)
    # Estimating the parameter `alpha` and `beta` as well
    Q21 = np.matrix([[0.01, 0.0, 0.0, 0.0, 0.0],[0.0, 0.01, 0.0, 0.0, 0.0],[0.0, 0.0, 0.01, 0.0, 0.0],[0.0, 0.0, 0.0, 0.01, 0.0],[0.0, 0.0, 0.0, 0.0, 0.01]])
    Q22 = np.matrix([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.01, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.01, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.01, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.01, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.01]])
    
    # Select one of the tuning matrices Q from above and P0 will be tuned accoridngly
    Q  = Q21
    P0 = 100*Q
    # The structure of matrix R stays unchanged (only temperature measurements available)
    R  = np.matrix([[0.01,0],[0,0.01]])
    
    "Note: Currently only a limited set of parameters is available for the following settings: "
    " type = continuous_discrete, as fully discret or continuous EKFs are not considered in the current implementation"
    " output_func = linear, while a fully nonlinear output model for the EKF will follow in the next release"
    " noise_level must be defined if the addition of a white-noise-like measurement is to be simulated, otherwise set to 0.0"
    setup_ekf = {
        'P0': P0,
        'Q' : Q,
        'R' : R,
        'C' : C,
        't_step': 0.005,
        'type': "continuous_discrete",
        'estimate_params': True,
        'output_func':'linear', 
        'noise_level':0.01
        #'output_func':'nonlinear',
        #'H_func': h
    }

    ekf.set_param(**setup_ekf)
    
    ekf.setup()
    
    return ekf
