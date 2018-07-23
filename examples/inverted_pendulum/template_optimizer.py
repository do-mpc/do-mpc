# 	 -*- coding: utf-8 -*-
#
#    This file is part of DO-MPC
#
#    DO-MPC: An environment for the easy, modular and efficient implementation of
#            robust nonlinear model predictive control
#
#    The MIT License (MIT)
#
#    Copyright (c) 2014-2018 Sergio Lucia, Alexandru Tatulea-Codrean
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

def optimizer(model):

    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """

    # Prediction horizon
    n_horizon = 20
    # Robust horizon, set to 0 for standard NMPC
    n_robust = 0
    # open_loop robust NMPC (1) or multi-stage NMPC (0). Only important if n_robust > 0
    open_loop = 0
    # Sampling time
    t_step = 0.1
    # Simulation time
    t_end = 10.0     # simulation time in minutes [min]
    # Choose type of state discretization (collocation or multiple-shooting)
    state_discretization = 'collocation'
    # Degree of interpolating polynomials: 1 to 5
    poly_degree = 2
    # Collocation points: 'legendre' or 'radau'
    collocation = 'radau'
    # Number of finite elements per control interval
    n_fin_elem = 2
    # NLP Solver and linear solver
    nlp_solver = 'ipopt'
    qp_solver = 'qpoases'

    # It is highly recommended that you use a more efficient linear solver
    # such as the hsl linear solver MA27, which can be downloaded as a precompiled
    # library and can be used by IPOPT on run time

    linear_solver = 'mumps'

    # GENERATE C CODE shared libraries NOTE: Not currently supported
    generate_code = 0

    """
    --------------------------------------------------------------------------
    template_optimizer: uncertain parameters
    --------------------------------------------------------------------------
    """
    # Define the different possible values of the uncertain parameters in the scenario tree
    alpha_values = NP.array([0.8, 1.0, 1.2])
    uncertainty_values = NP.array([alpha_values])
    """
    --------------------------------------------------------------------------
    template_optimizer: time-varying parameters
    --------------------------------------------------------------------------
    """
    # Only necessary if time-varying paramters defined in the model
    # The length of the vector for each parameter should be the prediction horizon
    # The vectos for each parameter might chance at each sampling time
    number_steps = int(t_end/t_step) + 1
    # Number of time-varying parameters
    n_tv_p = 1
    tv_p_values = NP.resize(NP.array([]),(number_steps,n_tv_p,n_horizon))
    for time_step in range (number_steps):
        if time_step < number_steps/2:
            tv_param_1_values = 2.5*NP.ones(n_horizon)
        else:
            tv_param_1_values = 1.0*NP.ones(n_horizon)
        
        tv_p_values[time_step] = NP.array([tv_param_1_values])
    # Parameteres of the NLP which may vary along the time (For example a set point that varies at a given time)
    set_point = SX.sym('set_point')
    parameters_nlp = NP.array([set_point])
    
    """
    --------------------------------------------------------------------------
    template_optimizer: pass_information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    # Check if the user has introduced the data correctly
    optimizer_dict = {'n_horizon':n_horizon, 'n_robust':n_robust, 't_step': t_step,
    't_end':t_end,'poly_degree': poly_degree, 'collocation':collocation,
    'n_fin_elem': n_fin_elem,'generate_code':generate_code,'open_loop': open_loop,
    'uncertainty_values':uncertainty_values,'parameters_nlp':parameters_nlp,
    'state_discretization':state_discretization,'nlp_solver': nlp_solver,
    'linear_solver':linear_solver, 'qp_solver':qp_solver, 'tv_p_values':tv_p_values}    
    optimizer_1 = core_do_mpc.optimizer(model,optimizer_dict)
    
    return optimizer_1
