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
    'linear_solver':linear_solver, 'qp_solver':qp_solver}
    optimizer_1 = core_do_mpc.optimizer(model,optimizer_dict)
    return optimizer_1
