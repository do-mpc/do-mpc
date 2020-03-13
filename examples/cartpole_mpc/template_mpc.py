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
        't_step': 0.02,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 1,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mpc.set_param(store_full_solution=True)

    _x, _u, _z, _tvp, p, _aux,  *_ = mpc.model.get_variables()
    #pdb.set_trace()
    mterm = (_x['x'] - 0.0)**2
    lterm = 10*(_x['theta'] - 0.0)**2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(F=0.001)

    mpc.bounds['lower', '_x', 'x'] = -10.0
    mpc.bounds['lower', '_x', 'v'] = -10.0
    mpc.bounds['lower', '_x', 'theta'] = -100*pi
    mpc.bounds['lower', '_x', 'omega'] = -10.0

    mpc.bounds['upper', '_x', 'x'] = 10.0
    mpc.bounds['upper', '_x', 'v'] = 10.0
    mpc.bounds['upper', '_x', 'theta'] = 100*pi
    mpc.bounds['upper', '_x', 'omega'] = 10.0
    
    mpc.bounds['lower', '_u', 'F'] = -1.0

    mpc.bounds['upper', '_u', 'F'] = 1.0


    # Instead of having a regular bound on T_R:
    #mpc.bounds['upper', '_x', 'T_R'] = 140
    # We can also have soft consraints as part of the set_nl_cons method:
    #mpc.set_nl_cons('x_pendulum', _aux['x_pedulum'], ub=5.0, soft_constraint=True, penalty_term_cons=1e2)


    mpc._x0['x'] = -1.0
    mpc._x0['v'] = 0.0
    mpc._x0['theta'] = 0.3
    mpc._x0['omega'] = 0.0

    D_var = np.array([0., -0.5, 0.5])

    mpc.set_uncertainty_values([D_var])

    mpc.setup()

    return mpc
