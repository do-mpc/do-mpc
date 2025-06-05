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

import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
 

def template_simulator(model):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """

    # initialisation of simulator
    simulator = do_mpc.simulator.Simulator(model)

    # modifying simulator settings
    simulator.set_param(t_step = 1)

    # Typically, the values would be reset at each call of p_fun.
    # Here we just return the fixed values:
    p_template = simulator.get_p_template()
    def p_fun(t_now):
        p_template['p1'] = 2
        return p_template
    simulator.set_p_fun(p_fun)

    # The timevarying paramters have no effect on the simulator (they are only part of the cost function).
    # We simply use the default values:
    tvp_template = simulator.get_tvp_template()
    def tvp_fun(t_now):
        if t_now<50:
            tvp_template['tvp1'] = 0.5
        else:
            tvp_template['tvp1'] = 1
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)


    # completing the simulator setup
    simulator.setup()

    # end of function
    return simulator
