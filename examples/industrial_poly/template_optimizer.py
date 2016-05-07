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

def template_optimizer(x,u,p):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    # Prediction horizon
    nk = 20

    # Sampling time
    t_step = 50.0/3600.0

    # Robust horizon, set to 0 for standard NMPC
    n_robust = 0

    # State discretization scheme: 'multiple-shooting' or 'collocation'
    state_discretization = 'collocation'

    # Collocation-specific options
    # Degree of interpolating polynomials: 1 to 5
    deg = 2
    # Collocation points: 'legendre' or 'radau'
    coll = 'radau'

    # Number of finite elements per control interval
    ni = 1

    # GENERATE C CODE shared libraries
    generate_code = 0

    # Simulate without feedback
    open_loop = 0

    # Simulation Time
    end_time = 12.0

    # NLP Solver and linear solver
    nlp_solver = 'ipopt'

    # It is highly recommended that you use a more efficient linear solver
    # such as the hsl linear solver MA27, which can be downloaded as a precompiled
    # library and can be used by IPOPT on run time

    linear_solver = 'ma27'


    """
    --------------------------------------------------------------------------
    template_optimizer: uncertain parameters
    --------------------------------------------------------------------------
    """
    # Define the different possible values of the uncertain parameters in the scenario tree
    k_0_values = NP.array([7.0*1.00, 7.0*1.30, 7.0*0.70])
    delH_R_values = NP.array([950.0, 950.0 * 1.30, 950.0 * 0.70])
    uncertainty_values = NP.array([delH_R_values, k_0_values])
    # Parameteres of the NLP which may vary along the time (For example a set point that varies at a given time)
    set_point = SX.sym('set_point')
    parameters_NLP = NP.array([set_point])

    """
    --------------------------------------------------------------------------
    template_optimizer: return data
    --------------------------------------------------------------------------
    """
    # Check if the user has introduced the data correctly

    return (nk, n_robust, t_step, end_time, deg, coll, ni, generate_code, +
            open_loop, uncertainty_values, parameters_NLP,  state_discretization, nlp_solver, linear_solver)
