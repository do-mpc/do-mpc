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


def template_mpc(model, w_ref, E_0, h_min=100):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 80,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.15,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)


    lterm = -model.aux['T_F']/1e4
    mpc.set_objective(mterm=DM(0), lterm=lterm)


    mpc.set_rterm(u_tilde=0.5)


    mpc.bounds['lower', '_x', 'theta'] = 0.0
    mpc.bounds['lower', '_x', 'phi'] = -0.5*np.pi
    mpc.bounds['lower', '_x', 'psi'] = -1.0*np.pi

    mpc.bounds['upper', '_x', 'theta'] = 0.5*np.pi
    mpc.bounds['upper', '_x', 'phi'] = 0.5*np.pi
    mpc.bounds['upper', '_x', 'psi'] = 1.0*np.pi

    mpc.bounds['lower','_u','u_tilde'] = -10
    mpc.bounds['upper','_u','u_tilde'] = 10

    h_min = h_min          # minimum height [m]

    mpc.set_nl_cons('height_kite',
        -model.aux['height_kite'],
        ub=-h_min,
        soft_constraint = True,
        penalty_term_cons = 1e3,
        maximum_violation = 10,
    )

    E_0_values = np.array([E_0])
    v_0_values = np.array([w_ref, w_ref*0.8, w_ref*1.2])

    mpc.set_uncertainty_values(E_0 = E_0_values, v_0 = v_0_values)
    mpc.setup()

    return mpc
