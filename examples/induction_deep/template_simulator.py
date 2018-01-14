#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2016 Sergio Lucia, Alexandru Tatulea-Codrean
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
#

from casadi import *
import numpy as NP
import core_do_mpc

def simulator(model):

    """
    --------------------------------------------------------------------------
    template_simulator: integration options
    --------------------------------------------------------------------------
    """
    # Choose the simulator time step
    t_step_simulator = 0.001
    # Choose options for the integrator
    opts = {"abstol":1e-10,"reltol":1e-10, "exact_jacobian":True, 'tf':t_step_simulator}
    # Choose integrator: for example 'cvodes' for ODEs or 'idas' for DAEs
    integration_tool = 'cvodes'

    # Choose the real value of the uncertain parameters that will be used
    # to perform the simulation of the system. They can be constant or time-varying
    def p_real_now(current_time):
        if current_time >= 0:
            p_real =  NP.array([1.15,1.15])
        else:
            p_real =  NP.array([1.0,1.0])
        return p_real

    # Choose the real value of the time-varing parameters
    def tv_p_real_now(current_time):
        tv_p_real = NP.array([1.0,1.0])
        return tv_p_real
    """
    --------------------------------------------------------------------------
    template_simulator: plotting options
    --------------------------------------------------------------------------
    """

    # Choose the indices of the states to plot
    plot_states = [0,1,2]
    # Choose the indices of the controls to plot
    plot_control = [0, 1]
    # Choose other varialbes to plot
    plot_other = [0, 1]
    # Plot animation (False or True)
    plot_anim = False
    # Export to matlab (for better plotting or postprocessing)
    export_to_matlab = True
    export_name = "mpc_result.mat"  # Change this name if desired

    """
    --------------------------------------------------------------------------
    template_simulator: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """

    simulator_dict = {'integration_tool':integration_tool,'plot_states':plot_states,
    'plot_control': plot_control, 'plot_other':plot_other,'plot_anim': plot_anim,'export_to_matlab': export_to_matlab,'export_name': export_name, 'p_real_now':p_real_now, 't_step_simulator': t_step_simulator, 'integrator_opts': opts, 'tv_p_real_now':tv_p_real_now}

    simulator_1 = core_do_mpc.simulator(model, simulator_dict)

    return simulator_1
