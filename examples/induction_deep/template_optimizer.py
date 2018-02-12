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


def optimizer(model):

    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """

    # Prediction horizon
    n_horizon = 10
    # Robust horizon, set to 0 for standard NMPC
    n_robust = 0
    # open_loop robust NMPC (1) or multi-stage NMPC (0). Only important if n_robust > 0
    open_loop = 0
    # Sampling time
    t_step = 1.0
    # Simulation time
    t_end = 40
    # Choose type of state discretization (collocation or multiple-shooting)
    state_discretization = 'collocation'
    # Degree of interpolating polynomials: 1 to 5
    poly_degree = 2
    # Collocation points: 'legendre' or 'radau'
    collocation = 'radau'
    # Number of finite elements per control interval
    n_fin_elem = 100
    # NLP Solver and linear solver
    nlp_solver = 'ipopt'
    qp_solver = 'qpoases'

    # It is highly recommended that you use a more efficient linear solver
    # such as the hsl linear solver MA27, which can be downloaded as a precompiled
    # library and can be used by IPOPT on run time

    linear_solver = 'ma27'

    # GENERATE C CODE shared libraries NOTE: Not currently supported
    generate_code = 0

    """
    --------------------------------------------------------------------------
    template_optimizer: uncertain parameters
    --------------------------------------------------------------------------
    """
    # Define the different possible values of the uncertain parameters in the scenario tree
    alpha_values = NP.array([1.0, 1.0, 1.0])
    beta_values = NP.array([1.0, 1.0, 1.0])
    uncertainty_values = NP.array([alpha_values,beta_values])
    # Parameteres of the NLP which may vary along the time (For example a set point that varies at a given time)
    set_point = SX.sym('set_point')
    parameters_nlp = NP.array([set_point])

    """
    --------------------------------------------------------------------------
    template_optimizer: time-varying parameters
    --------------------------------------------------------------------------
    """
    # Only necessary if time-varying paramters defined in the model
    # The length of the vector for each parameter should be the prediction horizon
    # The vectos for each parameter might chance at each sampling time
    number_steps = int(t_end/t_step*10000.0) + 1
    # Number of time-varying parameters
    n_tv_p = 2
    tv_p_values = NP.resize(NP.array([]),(number_steps,n_tv_p,n_horizon))
    for time_step in range (number_steps):
        if time_step < 4000:
            tv_param_1_values = 500 * NP.ones(n_horizon)
        elif time_step < 6000-10:
            tv_param_1_values = 3000 * NP.ones(n_horizon)
        else:
            tv_param_1_values = 1000 * NP.ones(n_horizon)
        tv_param_2_values = NP.tile(NP.array([1.0,0.0]),int(n_horizon/2))
        tv_p_values[time_step] = NP.array([tv_param_1_values,tv_param_2_values])
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
