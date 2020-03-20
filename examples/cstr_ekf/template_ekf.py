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
    ekf = do_mpc.estimator.EKF(model, ['K_1', 'K_2']) # ['K_1', 'K_2']
    
    
    C = np.matrix([[0,0,1,0],[0,0,0,1]])   # only temperatures are measured
    
    Q1 = np.matrix([[0.01, 0.0, 0.0, 0.0],[0.0, 0.01, 0.0, 0.0],[0.0, 0.0, 0.01, 0.0],[0.0, 0.0, 0.0, 0.01]])
    Q2 = np.matrix([[0.1, 0.0, 0.0, 0.0],[0.0, 0.1, 0.0, 0.0],[0.0, 0.0, 0.2, 0.0],[0.0, 0.0, 0.0, 0.5]])       #  Different tunning
    Q  = Q2
    P0 = 100*Q
    R  = np.matrix([[0.01,0],[0,0.01]])
    
    setup_ekf = {
        'P0': P0,
        'Q': Q,
        'R': R,
        't_step': 0.005,
        'type': "continuous_discrete",
        'output_func':'linear', 
        'C_mat': C,
        'x_hat': [0, 0, 0, 0],
        'noise_level':0.2
        #'output_func':'nonlinear',
        #'H_func': h
    }

    ekf.set_param(**setup_ekf)
    
    ekf.setup()
    
    return ekf
