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


def template_mhe(model):
    """
    --------------------------------------------------------------------------
    template_mhe: tuning parameters
    --------------------------------------------------------------------------
    """
    mhe = do_mpc.estimator.MHE(model, ['Theta_1'])

    setup_mhe = {
        'n_horizon': 10,
        't_step': 0.1,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'},
    }

    mhe.set_param(**setup_mhe)

    # MHE cost function:
    y_meas = mhe._y_meas
    y_calc = mhe._y_calc

    dy = y_meas.cat-y_calc.cat
    P_y = np.diag(np.array([1,1,1,20,20]))

    stage_cost = dy.T@P_y@dy

    x_0 = mhe._x
    x_prev = mhe._x_prev
    p_0 = mhe._p_est
    p_prev = mhe._p_est_prev

    dx = x_0.cat - x_prev.cat
    dp = p_0.cat - p_prev.cat

    P_x = np.eye(8)
    P_p = np.eye(1)

    arrival_cost = 1e-4*dx.T@P_x@dx + 1e0*dp.T@P_p@dp

    mhe.set_objective(stage_cost, arrival_cost)


    # The timevarying paramters have no effect on the simulator (they are only part of the cost function).
    # We simply use the default values:
    tvp_template = mhe.get_tvp_template()
    def tvp_fun(t_now):
        return tvp_template
    mhe.set_tvp_fun(tvp_fun)


    # Only the non estimated parameters must be passed:
    p_template_mhe = mhe.get_p_template()
    def p_fun_mhe(t_now):
        p_template_mhe['Theta_2'] = 2.25e-4
        p_template_mhe['Theta_3'] = 2.25e-4
        return p_template_mhe
    mhe.set_p_fun(p_fun_mhe)

    # Measurement function:
    y_template = mhe.get_y_template()

    def y_fun(t_now):
        n_steps = min(mhe.data._y.shape[0], mhe.n_horizon)
        for k in range(-n_steps,0):
            y_template['y_meas',k] = mhe.data._y[k]

        return y_template

    mhe.set_y_fun(y_fun)

    mhe.bounds['lower','_u','phi_m_set'] = -5
    mhe.bounds['upper','_u','phi_m_set'] = 5

    mhe.bounds['lower','_p_est', 'Theta_1'] = 1e-5
    mhe.bounds['upper','_p_est', 'Theta_1'] = 1e-3

    mhe.setup()

    return mhe
