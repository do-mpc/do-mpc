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

def setup_solver():

    # Call template_model to calculate the model
    model_1 = template_model()
    #x, u, xdot, p, z, x0, x_lb, x_ub, u0, u_lb, u_ub, x_scaling, u_scaling, cons, cons_ub, cons_terminal, cons_terminal_lb, cons_terminal_ub, soft_constraint, penalty_term_cons, maximum_violation, mterm, lterm, rterm = template_model()
    # Call template_optimizer to get the OCP formulation
    # TODO: Check names of the mode. Maybe include in do_mpc configuration
    optimizer_1 = template_optimizer(model_1)
    configuration_1 = core_do_mpc.do_mpc_configuration(model_1, optimizer_1, [],[])
    #nk, n_robust, t_step, end_time, deg, coll, ni, generate_code, open_loop, uncertainty_values, parameters_NLP, state_discretization, nlp_solver, linear_solver = template_optimizer(x,u,p)

    # Call setup_nlp to generate the NLP
    nlp_dict_out = setup_nlp(configuration_1.model, configuration_1.optimizer)

    # nlp_fcn, X_offset, U_offset, E_offset, vars_lb, vars_ub, vars_init, lbg, ubg, parent_scenario, child_scenario, n_branches, n_scenarios =  setup_nlp(nk, n_robust, t_step, end_time, deg, coll, ni, generate_code,
    # open_loop, uncertainty_values, parameters_NLP, x0, x_lb, x_ub,
    # u0, u_lb, u_ub, x_scaling, u_scaling, cons, cons_ub,
    # cons_terminal, cons_terminal_lb, cons_terminal_ub,
    # soft_constraint, penalty_term_cons, maximum_violation,
    # mterm, lterm, rterm, state_discretization, x, xdot, u, p)

    # Set options
    opts = {}
    opts["expand"] = True
    opts["ipopt.linear_solver"] = configuration_1.optimizer.linear_solver
    #TODO: this should be passed as parameters of the optimizer class
    opts["ipopt.max_iter"] = 500
    opts["ipopt.tol"] = 1e-6
    # Setup the solver
    solver = nlpsol("solver", configuration_1.optimizer.nlp_solver, nlp_dict_out['nlp_fcn'], opts)
    arg = {}

    # Initial condition
    arg["x0"] = nlp_dict_out['vars_init']

    # Bounds on x
    arg["lbx"] = nlp_dict_out['vars_lb']
    arg["ubx"] = nlp_dict_out['vars_ub']

    # Bounds on g
    arg["lbg"] = nlp_dict_out['lbg']
    arg["ubg"] = nlp_dict_out['ubg']
    # NLP parameters
    arg["p"] = configuration_1.model.ocp.u0
    # TODO: better way than adding new fields here?
    configuration_1.optimizer.solver = solver
    configuration_1.optimizer.arg = arg

    return configuration_1, nlp_dict_out
