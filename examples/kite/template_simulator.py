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
from wind_model import Wind


def template_simulator(model, w_ref, E_0):
    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    # setting up parameters for the simulator
    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.15
    }
    simulator.set_param(**params_simulator)

    # setting up parameters for the simulator
    p_num = simulator.get_p_template()

    # Get instance of wind class and set mean and timestep:
    wind = Wind(w_ref = w_ref, t_step = simulator.settings.t_step, k_sigma_w=0.2)

    # E_0 is now a random walk parameter (clipped if it deviates too much)
    simulator.E_0 = E_0
    E_0_min = 0.8*E_0
    E_0_max = 1.2*E_0

    def p_fun(t_now):
        v_0 = wind.make_step()
        simulator.E_0 += 0.01*np.random.randn()
        simulator.E_0 = np.clip(simulator.E_0, E_0_min, E_0_max)
        p_num['E_0'] = simulator.E_0
        p_num['v_0'] = v_0
        return p_num
    simulator.set_p_fun(p_fun)

    # completing the simulator setup
    simulator.setup()

    # end of function
    return simulator
