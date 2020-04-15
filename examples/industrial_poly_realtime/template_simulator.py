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
from opcmodules import RealtimeSimulator, RealtimeController

def template_simulator(model, opc_opts):
    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """    
    # The simulator is the one that typically run the fastest (most often, e.g every second)
    opc_opts['_opc_opts']['_client_type'] = "simulator"
    opc_opts['_cycle_time'] = 2.0
    simulator = RealtimeSimulator(model,opc_opts)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 2.0/3600.0
    }

    simulator.set_param(**params_simulator)

    p_num = simulator.get_p_template()
    p_num['delH_R'] = 950.0
    p_num['k_0'] = 7.0
    def p_fun(t_now):
        return p_num
    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator
