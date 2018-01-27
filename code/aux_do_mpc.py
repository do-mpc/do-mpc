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
import pdb


# TODO: here also add the automatic checking of the collocation accuracy and the calculation of a good initial condition
def get_good_initialization(configuration):
    "This function uses the current initial state and the previous, shifted, control trajectory to obtain a good initialization for all the collocation points"
    # TODO: improve by shifting the predicted trajectory and taking different inputs instead of a constant one

    # Get necessary values
    # NV = configuration.optimizer.nlp_dict_out['NV']
    X_offset = configuration.optimizer.nlp_dict_out['X_offset']
    n_scenarios = configuration.optimizer.nlp_dict_out['n_scenarios']
    n_branches  = configuration.optimizer.nlp_dict_out['n_branches']
    branch_offset  = configuration.optimizer.nlp_dict_out['branch_offset']
    p_scenario  = configuration.optimizer.nlp_dict_out['p_scenario']
    tv_p_values = configuration.optimizer.tv_p_values
    nk = configuration.optimizer.n_horizon
    n_robust = configuration.optimizer.n_robust
    t_step = configuration.optimizer.t_step
    deg = configuration.optimizer.poly_degree
    coll = configuration.optimizer.collocation
    ni = configuration.optimizer.n_fin_elem

    nx = configuration.model.x.size(1)
    nu = configuration.model.u.size(1)
    np = configuration.model.p.size(1)
    nz = configuration.model.z.size(1)
    # Use as initial values the ones that have been provided
    vars_init = vertcat(configuration.optimizer.arg["x0"])
    t0_original = configuration.simulator.t0_sim
    t0_initialization = configuration.simulator.t0_sim
    tf_initialization = t0_initialization + t_step/ni/(deg+1)
    x0_initialization = configuration.simulator.x0_sim
    u_initialization  = configuration.optimizer.u_mpc
    vars_init[X_offset[0,0]  :  X_offset[0,0] + nx] = vertcat(x0_initialization)

    # Initialize the time with the correct step based on the collocaiton points
    configuration.simulator.t0_sim = t0_initialization
    configuration.simulator.tf_sim = tf_initialization
    # Skip the initialization for  the initial condition
    offset = nx
    # Loop over prediction horizon (stage), nodes in the current stage,
    # children nodes for each node, finite elements and collocation points
    for k in range(nk):
        for s in range(n_scenarios[k]):
            for b in range(n_branches[k]):
              #first_j = 1
                for i in range(ni):
                    for j in range(deg+1):
                        # integrate
                        # pdb.set_trace()
                        result  = configuration.simulator.simulator(x0 = x0_initialization[0:nx], p = vertcat(u_initialization,p_scenario[b+branch_offset[k][s]],tv_p_values[k,:,k]))
                        x_next = NP.squeeze(vertcat(NP.squeeze(result['xf']),NP.squeeze(result['zf'])))
                        #pdb.set_trace()
                        vars_init[X_offset[k,s] + offset   :  X_offset[k,s] + nx + offset]   = x_next
                        offset += nx
                        t0_initialization = tf_initialization
                        tf_initialization = tf_initialization + t_step/ni/(deg+1)
                        x0_initialization = x_next
                vars_init[X_offset[k,s] + offset : X_offset[k,s] + nx + offset] = x_next
                offset +=nx
            offset = 0
    for  s in range(n_scenarios[nk]):
        vars_init[X_offset[nk,s] + offset   :  X_offset[nk,s] + nx + offset]   = x_next

    # Update the initial guess for the solver
    configuration.optimizer.arg['x0'] = vars_init
    # Restore the initial simulation times
    configuration.simulator.t0_sim = t0_original
    configuration.simulator.tf_sim = t0_original + configuration.simulator.t_step_simulator

def check_collocation_accuracy(configuration):
    "This function checks the accuracy of the chosen orthogonal collocation scheme when compared to an integration with SUNDIALS using tight tolerances"
    X_offset = configuration.optimizer.nlp_dict_out['X_offset']
    U_offset = configuration.optimizer.nlp_dict_out['U_offset']
    n_scenarios = configuration.optimizer.nlp_dict_out['n_scenarios']
    n_branches  = configuration.optimizer.nlp_dict_out['n_branches']
    branch_offset  = configuration.optimizer.nlp_dict_out['branch_offset']
    p_scenario  = configuration.optimizer.nlp_dict_out['p_scenario']
    tv_p_values = configuration.optimizer.tv_p_values
    nk = configuration.optimizer.n_horizon
    n_robust = configuration.optimizer.n_robust
    t_step = configuration.optimizer.t_step
    deg = configuration.optimizer.poly_degree
    coll = configuration.optimizer.collocation
    ni = configuration.optimizer.n_fin_elem

    nx = configuration.model.x.size(1)
    nu = configuration.model.u.size(1)
    np = configuration.model.p.size(1)
    nz = configuration.model.z.size(1)
    # Use as initial values the ones that have been provided
    vars_init = vertcat(configuration.optimizer.arg["x0"])
    t0_original = configuration.simulator.t0_sim
    t0_initialization = configuration.simulator.t0_sim
    tf_initialization = t0_initialization + t_step/ni/(deg+1)
    x0_initialization = configuration.simulator.x0_sim
    u_initialization  = configuration.optimizer.u_mpc
    vars_init[X_offset[0,0]  :  X_offset[0,0] + nx] = vertcat(x0_initialization)

    # Initialize the time with the correct step based on the collocaiton points

    # Skip the initialization for  the initial condition
    offset = X_offset[0,0]
    # Optimal solution of the optimizer
    # v_opt = configuration.optimizer.opt_result_step.optimal_solution
    # NOTE this has been changed to run it in the first iteration
    v_opt = configuration.optimizer.arg["x0"]
    error_coll = NP.resize(NP.array([]),(v_opt.shape))

    # Loop over prediction horizon (stage), nodes in the current stage,
    # children nodes for each node, finite elements and collocation points
    model_simulator = configuration.model
    dae = {'x':model_simulator.x, 'p':vertcat(model_simulator.u,model_simulator.p,model_simulator.tv_p), 'ode':model_simulator.ode}
    # Choose collocation points
    coll = configuration.optimizer.collocation
    if coll=='legendre':    # Legendre collocation points
        tau_root = [0]+collocation_points(deg, 'legendre')
    elif coll=='radau':     # Radau collocation points
        tau_root = [0]+collocation_points(deg, 'radau')
    else:
        raise Exception('Unknown collocation scheme')
    h = 1.0/ni

    # All collocation time points
    T = NP.zeros((nk,ni,deg+1))
    for k in range(nk):
        for i in range(ni):
            for j in range(deg+1):
                T[k,i,j] = h*(k*ni + i + tau_root[j])
    # pdb.set_trace()
    for k in range(nk):
        for s in range(n_scenarios[k]):
            for b in range(n_branches[k]):
              #first_j = 1
                u_initialization = v_opt[U_offset[k,s]:U_offset[k,s]+nu]
                # add initial point
                vars_init[offset : nx + offset] = x0_initialization
                offset +=nx
                for i in range(ni):
                    for j in range(deg+1):
                        # integrate

                        # add initial point
                        T[k,i,j] = h*(k*ni + i + tau_root[j])
                        t_now = T[k,i,j]
                        if (j == deg and i == ni -1): # if last point of last element
                            t_next = t_now # do not do quadrature here
                        else:
                            if (j == deg and i < ni - 1):
                                t_next = T[k,i+1,0]
                            else:
                                t_next = T[k,i,j+1]
                        configuration.simulator.t0_sim = t_now
                        configuration.simulator.tf_sim = t_next
                        opts = {"abstol":1e-12,"reltol":1e-12, "exact_jacobian":True, 'tf':t_next-t_now}
                        configuration.simulator.simulator2 = integrator("simulator", "cvodes", dae,  opts)
                        result  = configuration.simulator.simulator2(x0 = x0_initialization[0:nx], p = vertcat(u_initialization,p_scenario[b+branch_offset[k][s]],tv_p_values[k,:,k]))
                        x_next = NP.squeeze(vertcat(NP.squeeze(result['xf']),NP.squeeze(result['zf'])))
                        # pdb.set_trace()
                        vars_init[offset : nx + offset]   = x_next
                        offset += nx
                        x0_initialization = x_next
                    # vars_init[offset : nx + offset] = x_next
                    # offset +=nx
            vars_init[offset:offset+nu] = u_initialization
            offset += nu
    vars_init[offset : nx + offset] = x_next
    offset +=nx
            # offset = 0
    # for  s in range(n_scenarios[nk]):
    #     vars_init[X_offset[nk,s] + offset   :  X_offset[nk,s] + nx + offset]   = x_next

    # Update the initial guess for the solver
    error_coll = vars_init - v_opt
    configuration.simulator.error_coll = error_coll
    configuration.optimizer.arg['x0'] = vars_init
    # Restore the initial simulation times
    configuration.simulator.t0_sim = t0_original
    configuration.simulator.tf_sim = t0_original + configuration.simulator.t_step_simulator
