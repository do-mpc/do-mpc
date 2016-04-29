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


def template_simulator(model):
    # Here you can use a different model for the simulator if desired
    #x, u, xdot, p, z, x0, x_lb, x_ub, u0, u_lb, u_ub, x_scaling, u_scaling, #cons, cons_ub, cons_terminal, cons_terminal_lb, cons_terminal_ub, #soft_constraint, penalty_term_cons, maximum_violation, mterm, lterm, rterm = #template_model()
    #xdot = substitute(xdot,x,x*x_scaling)/x_scaling
    #xdot = substitute(xdot,u,u*u_scaling)
    #up = vertcat(u,p)

    #f_sim = Function("f_sim", controldaeIn(x=x,p=p,u=u),daeOut(ode=xdot))
    #f_sim = Function("f_sim", [x,p,u],[xdot])
    # TODO: should we allow for a different t_step compared to the optimizer t_step? In general is possible to have a smaller simulation t_step, but this might cause errors if people forget to change it
    t_step = 0.005
    opts = {}
    #opts["integrator"] = "cvodes"
    # FIXME Only one step integration
    opts = {"abstol":1e-10,"reltol":1e-10, "exact_jacobian":True, 'tf':t_step}
    #N = 2
    # Use integrator: for example 'cvodes' for ODEs or 'idas' for DAEs
    integrator_tool = 'cvodes'
    #dae = {'x':x, 'p':vertcat(u,p), 'ode':xdot}
    #tgrid = linspace(0,t_step,N)
    #simulator = integrator("simulator", "cvodes", dae,  opts)
    #return simulator

#def plotting_options():
    # Choose the indices of the states to plot
    plot_states = [0, 1, 2]
    # Choose the indices of the controls to plot
    plot_control = [0, 1]
    # Plot animation (False or True)
    plot_anim = False
    # Export to matlab (for better plotting or postprocessing)
    export_to_matlab = True
    export_name = "mpc_result.mat"  # Change this name if desired
    #return plot_states, plot_control, plot_anim, export_to_matlab, export_name

#def real_parameters(current_time):
    # Here choose the real value of the uncertain parameters that will be chosen
    # to perform the simulation of the system. They can be constant or time-varying
    p_real =  NP.array([1.0,1.0])

    simulator_dict = {'integration_tool':integration_tool,'plot_states':plot_states,
    'plot_control': plot_control,'plot_anim': plot_anim,'export_to_matlab': export_to_matlab,'export_name': export_name, 'p_real':p_real}

    simulator_1 = core_do_mpc.simulator(model_1, simulator_dict)

    return simulator_1
