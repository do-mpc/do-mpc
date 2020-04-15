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
import do_mpc
from opcmodules import RealtimeEstimator

def template_estimator(model, opc_opts):
    """
    --------------------------------------------------------------------------
    template_estimator: no parameters to pass, this is a "mock" state estimator
    --------------------------------------------------------------------------
    """    
    # The estimator is just a delayed state feedback estimator in this case 

    opc_opts['_opc_opts']['_client_type'] = "estimator"
    opc_opts['_cycle_time'] = 3.0   # execute every 3 seconds
    opc_opts['_opc_opts']['_output_feedback'] = True

    # Use type 'SFB' for a full state feedback or 'EKF'/'MHE' for an actual estimator
    etype = 'SFB'
    if etype == 'SFB':
        estimator = RealtimeEstimator('SFB', model, opc_opts)
    if etype == 'EKF':
        estimator = RealtimeEstimator('EKF', model, opc_opts)
        
        # Tuning paramaters EKF
        # x    = [m_W, m_A, m_P, T_R, T_S, Tout_M, T_EK, T_AWT, accum_monom, T_adiab]
        # meas =   |    0    0    |    |      |     |     |          |        0
        
        C = np.matrix([[1,0,0,0,0,0,0,0,0,0],
                       [0,0,0,1,0,0,0,0,0,0],
                       [0,0,0,0,1,0,0,0,0,0],
                       [0,0,0,0,0,1,0,0,0,0],
                       [0,0,0,0,0,0,1,0,0,0],
                       [0,0,0,0,0,0,0,1,0,0],
                       [0,0,0,0,0,0,0,0,1,0]])  
        
        Q = 0.001*np.eye(10)
        R = 0.01*np.eye(7)
        P0 = 100*Q
        
        setup_ekf = {
        'P0': P0,
        'Q' : Q,
        'R' : R,
        'C' : C,
        't_step': 3.0/3600.0,
        'type': "continuous_discrete",
        'estimate_params': False,
        'output_func':'linear', 
        'noise_level':0.01
        #'output_func':'nonlinear',
        #'H_func': h
        }

        estimator.set_param(**setup_ekf) 
        estimator.setup()
        
    return estimator
