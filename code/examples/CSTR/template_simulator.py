# 	 -*- coding: utf-8 -*-
#
#    This file is part of DO-MPC
#
#    DO-MPC: An environment for the easy, modular and efficient implementation of
#            robust nonlinear model predictive control
#
#    The MIT License (MIT)
#
#    Copyright (c) 2014-2015 Sergio Lucia, Alexandru Tatulea-Codrean, Sebastian Engell
#                            TU Dortmund. All rights reserved
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
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
    # TODO: should we allow for a different t_step compared to the optimizer t_step? In general is possible to have a smaller simulation t_step, but this might cause errors if people forget to change it
    t_step_simulator = 0.005
    opts = {}
    # FIXME integration not valid for time-varying systems
    opts = {"abstol":1e-10,"reltol":1e-10, "exact_jacobian":True, 'tf':t_step_simulator}
    #N = 2
    # Use integrator: for example 'cvodes' for ODEs or 'idas' for DAEs
    integration_tool = 'cvodes'

    # Here choose the real value of the uncertain parameters that will be chosen
    # to perform the simulation of the system. They can be constant or time-varying
    def p_real_now(current_time):
        if current_time >= 0:
            p_real =  NP.array([1.0,1.0])
        else:
            p_real =  NP.array([1.0,1.0])
        return p_real
    """
    --------------------------------------------------------------------------
    template_simulator: plotting options
    --------------------------------------------------------------------------
    """
    # Choose the indices of the states to plot
    plot_states = [1,1,2]
    # Choose the indices of the controls to plot
    plot_control = [0, 1]
    # Plot animation (False or True)
    plot_anim = False
    # Export to matlab (for better plotting or postprocessing)
    export_to_matlab = False
    export_name = "mpc_result.mat"  # Change this name if desired




    """
    --------------------------------------------------------------------------
    template_simulator: pass information
    --------------------------------------------------------------------------
    """
    simulator_dict = {'integration_tool':integration_tool,'plot_states':plot_states,
    'plot_control': plot_control,'plot_anim': plot_anim,'export_to_matlab': export_to_matlab,'export_name': export_name, 'p_real_now':p_real_now, 't_step_simulator': t_step_simulator, 'integrator_opts': opts}

    simulator_1 = core_do_mpc.simulator(model, simulator_dict)

    return simulator_1
