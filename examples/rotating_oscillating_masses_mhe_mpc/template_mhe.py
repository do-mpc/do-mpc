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


def template_mhe(model, silence_solver = False):
    """
    --------------------------------------------------------------------------
    template_mhe: tuning parameters
    --------------------------------------------------------------------------
    """
    mhe = do_mpc.estimator.MHE(model, ['Theta_1'])

    # settings for mhe
    mhe.settings.n_horizon =  10
    mhe.settings.t_step =  0.1
    mhe.settings.store_full_solution =  True
    mhe.settings.nl_cons_check_colloc_points =  True

    # suppress solver output
    if silence_solver:
        mhe.settings.supress_ipopt_output()

    # Set the default MHE objective by passing the weighting matrices:
    P_v = model.tvp['P_v']
    P_x = 1e-4*np.eye(8)
    P_p = model.p['P_p']
    mhe.set_default_objective(P_x, P_v, P_p)

    # P_y is listed in the time-varying parameters and must be set.
    # This is more of a proof of concept (P_y is not actually changing over time).
    # We therefore do the following:
    tvp_template = mhe.get_tvp_template()
    tvp_template['_tvp', :, 'P_v'] = np.diag(np.array([1,1,1,20,20]))
    # Typically, the values would be reset at each call of tvp_fun.
    # Here we just return the fixed values:
    def tvp_fun(t_now):
        return tvp_template
    mhe.set_tvp_fun(tvp_fun)


    # Only the non estimated parameters must be passed:
    p_template_mhe = mhe.get_p_template()
    def p_fun_mhe(t_now):
        p_template_mhe['Theta_2'] = 2.25e-4
        p_template_mhe['Theta_3'] = 2.25e-4
        # And our previously set P_x:
        p_template_mhe['P_p'] = np.eye(1)
        return p_template_mhe
    mhe.set_p_fun(p_fun_mhe)

    # Measurement function:
    y_template = mhe.get_y_template()

    def y_fun(t_now):
        n_steps = min(mhe.data._y.shape[0], mhe.settings.n_horizon)
        for k in range(-n_steps,0):
            y_template['y_meas',k] = mhe.data._y[k]

        return y_template

    mhe.set_y_fun(y_fun)

    # setting up boundaries for the inputs
    mhe.bounds['lower','_u','phi_m_set'] = -5
    mhe.bounds['upper','_u','phi_m_set'] = 5

    # setting up boundaries for the states
    mhe.bounds['lower','_x', 'dphi'] = -6
    mhe.bounds['upper','_x', 'dphi'] = 6

    # Instead of setting bound like this:
    # mhe.bounds['lower','_p_est', 'Theta_1'] = 1e-5
    # mhe.bounds['upper','_p_est', 'Theta_1'] = 1e-3

    # The MHE also supports nonlinear constraints (here they are still linear however) ...
    mhe.set_nl_cons('p_est_lb', -mhe._p_est['Theta_1']+1e-5, 0)
    mhe.set_nl_cons('p_est_ub', mhe._p_est['Theta_1']-1e-3, 0)

    # completing the setup of the mpc
    mhe.setup()

    # end of function
    return mhe
