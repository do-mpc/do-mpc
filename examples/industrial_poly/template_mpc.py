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


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 20,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 50.0/3600.0,
        'state_discretization': 'collocation',
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    _x = model.x

    mterm = - _x['m_P']
    lterm = - _x['m_P']

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(m_dot_f=0.002, T_in_M=0.004, T_in_EK=0.002)

    temp_range = 2.0

    mpc.bounds['lower','_x','m_W'] = 0.0
    mpc.bounds['lower','_x','m_A'] = 0.0
    mpc.bounds['lower','_x','m_P'] = 26.0

    mpc.bounds['lower','_x','T_R'] = 363.15 - temp_range
    mpc.bounds['lower','_x','T_S'] = 298.0
    mpc.bounds['lower','_x','Tout_M'] = 298.0
    mpc.bounds['lower','_x','T_EK'] = 288.0
    mpc.bounds['lower','_x','Tout_AWT'] = 288.0
    mpc.bounds['lower','_x','accum_monom'] = 0.0

    mpc.bounds['upper','_x','T_R'] = 363.15 + temp_range
    mpc.bounds['upper','_x','T_S'] = 400.0
    mpc.bounds['upper','_x','Tout_M'] = 400.0
    mpc.bounds['upper','_x','T_EK'] = 400.0
    mpc.bounds['upper','_x','Tout_AWT'] = 400.0
    mpc.bounds['upper','_x','accum_monom'] = 30000.0
    mpc.bounds['upper','_x','T_adiab'] = 382.15 + 10.0

    mpc.bounds['lower','_u','m_dot_f'] = 0.0
    mpc.bounds['lower','_u','T_in_M'] = 333.15
    mpc.bounds['lower','_u','T_in_EK'] = 333.15

    mpc.bounds['upper','_u','m_dot_f'] = 3.0e4
    mpc.bounds['upper','_u','T_in_M'] = 373.15
    mpc.bounds['upper','_u','T_in_EK'] = 373.15

    # Scaling
    mpc.scaling['_x','m_W'] = 10
    mpc.scaling['_x','m_A'] = 10
    mpc.scaling['_x','m_P'] = 10
    mpc.scaling['_x','accum_monom'] = 10

    mpc.scaling['_u','m_dot_f'] = 100


    delH_R_var = np.array([950.0, 950.0 * 1.30, 950.0 * 0.70])
    k_0_var = np.array([7.0*1.00, 7.0*1.30, 7.0*0.70])
    mpc.set_uncertainty_values([delH_R_var, k_0_var])
    
    # Instead of having a regular bound on T_adiab:
    #mpc.bounds['upper', '_x', 'T_adiab'] = 382.15
    # We can also have soft consraints as part of the set_nl_cons method:
    mpc.set_nl_cons('T_adiab', _x['T_adiab'], ub=381.0, soft_constraint=True, penalty_term_cons=1e4)



    mpc.setup()

    return mpc
