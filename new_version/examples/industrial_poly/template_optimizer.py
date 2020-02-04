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


def template_optimizer(model):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    optimizer = do_mpc.optimizer(model)

    setup_optimizer = {
        'n_horizon': 20,
        'n_robust': 1,
        'open_loop': 0,
        't_step': 50.0/3600.0,
        'state_discretization': 'collocation',
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'},
        'store_full_solution': True,
    }

    optimizer.set_param(**setup_optimizer)

    _x, _u, _z, _tvp, p, _aux,  *_ = optimizer.model.get_variables()

    mterm = - _x['m_P']
    lterm = - _x['m_P']

    optimizer.set_objective(mterm=mterm, lterm=lterm)
    optimizer.set_rterm(m_dot_f=0.002, T_in_M=0.004, T_in_EK=0.002)

    temp_range = 2.0

    optimizer.bounds['lower','_x','m_W'] = 0.0
    optimizer.bounds['lower','_x','m_A'] = 0.0
    optimizer.bounds['lower','_x','m_P'] = 26.0

    optimizer.bounds['lower','_x','T_R'] = 363.15 - temp_range
    optimizer.bounds['lower','_x','T_S'] = 298.0
    optimizer.bounds['lower','_x','Tout_M'] = 298.0
    optimizer.bounds['lower','_x','T_EK'] = 288.0
    optimizer.bounds['lower','_x','Tout_AWT'] = 288.0
    optimizer.bounds['lower','_x','accum_monom'] = 0.0

    optimizer.bounds['upper','_x','T_R'] = 363.15 + temp_range + 10.0
    optimizer.bounds['upper','_x','T_S'] = 400.0
    optimizer.bounds['upper','_x','Tout_M'] = 400.0
    optimizer.bounds['upper','_x','T_EK'] = 400.0
    optimizer.bounds['upper','_x','Tout_AWT'] = 400.0
    optimizer.bounds['upper','_x','accum_monom'] = 30000.0
    optimizer.bounds['upper','_x','T_adiab'] = 382.15 + 10.0

    optimizer.bounds['lower','_u','m_dot_f'] = 0.0
    optimizer.bounds['lower','_u','T_in_M'] = 333.15
    optimizer.bounds['lower','_u','T_in_EK'] = 333.15

    optimizer.bounds['upper','_u','m_dot_f'] = 3.0e4
    optimizer.bounds['upper','_u','T_in_M'] = 373.15
    optimizer.bounds['upper','_u','T_in_EK'] = 373.15

    # Scaling
    optimizer.scaling['_x','m_W'] = 10
    optimizer.scaling['_x','m_A'] = 10
    optimizer.scaling['_x','m_P'] = 10
    optimizer.scaling['_x','accum_monom'] = 10

    optimizer.scaling['_u','m_dot_f'] = 100


    delH_R_var = np.array([950.0, 950.0 * 1.30, 950.0 * 0.70])
    k_0_var = np.array([7.0*1.00, 7.0*1.30, 7.0*0.70])
    optimizer.set_uncertainty_values([delH_R_var, k_0_var])

    optimizer.setup()

    return optimizer
