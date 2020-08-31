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
        'n_robust': 0,
        'n_horizon': 20,
        't_step': 0.1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    _x, _tvp  = model['x', 'tvp']

    lterm = (_x['phi_2'] - _tvp['phi_2_set'])**2
    mterm = DM(1)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(phi_m_set=1e-2)


    # Create an interesting trajectory for the setpoint (_tvp)
    # by randomly choosing a new value or keeping the previous one.
    def random_setpoint(tvp0):
        tvp_next = (0.5-np.random.rand(1))*np.pi
        switch = np.random.rand() >= 0.95
        tvp0 = (1-switch)*tvp0 + switch*tvp_next
        return tvp0

    np.random.seed(999)
    tvp_traj = [np.array([0])]
    for i in range(400):
        tvp_traj.append(random_setpoint(tvp_traj[i]))

    tvp_traj = np.concatenate(tvp_traj)

    # Create tvp_fun that takes element from that previously defined trajectory
    # depending on the current timestep.
    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_now):
        ind = int(t_now/setup_mpc['t_step'])
        tvp_template['_tvp', :-1] = vertsplit(tvp_traj[ind:ind+setup_mpc['n_horizon']])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    inertia_mass_1 = 2.25*1e-4*np.array([1.,1.1])
    inertia_mass_2 = 2.25*1e-4*np.array([1.,])
    inertia_mass_3 = 2.25*1e-4*np.array([1.])

    mpc.set_uncertainty_values(
        Theta_1 = inertia_mass_1,
        Theta_2 = inertia_mass_2,
        Theta_3 = inertia_mass_3,)

    mpc.bounds['lower','_u','phi_m_set'] = -5
    mpc.bounds['upper','_u','phi_m_set'] = 5

    mpc.setup()

    return mpc
