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
    x, u, xdot, p, z, x0, x_lb, x_ub, u0, u_lb, u_ub, x_scaling, u_scaling, cons, cons_lb, cons_ub, soft_constraint, penalty_term_cons, maximum_violation_ub, maximum_violation_lb, mterm, lterm, rterm = template_model()
    # Call template_optimizer to get the OCP formulation
    nk, n_robust, t_step, end_time, deg, coll, ni, generate_code, open_loop, uncertainty_values, parameters_NLP, state_discretization = template_optimizer(x,u,p)    
	
    # Call setup_nlp to generate the NLP
    nlp_fcn, X_offset, U_offset, E_offset, vars_lb, vars_ub, vars_init, lbg, ubg, parent_scenario, child_scenario, n_branches, n_scenarios =  setup_nlp(nk, n_robust, t_step, end_time, deg, coll, ni, generate_code, 
                                                                                    open_loop, uncertainty_values, parameters_NLP, x0, x_lb, x_ub, 
                                                                                    u0, u_lb, u_ub, x_scaling, u_scaling, cons, cons_lb, cons_ub, 
                                                                                    soft_constraint, penalty_term_cons, maximum_violation_ub, 
                                                                                    maximum_violation_lb, mterm, lterm, rterm, state_discretization, x, xdot, u, p)
    
    # Setup the solver    
    solver = NlpSolver('ipopt',nlp_fcn)
    # Set options
    #solver.setOption("verbose",True)
    if state_discretization=='collocation':
    	solver.setOption("expand",True)
    solver.setOption("max_iter",200)
    solver.setOption("tol",1e-6)
    solver.setOption("linear_solver","ma27")
    
    # initialize the solver
    solver.init()
      
    # Initial condition
    solver.setInput(vars_init,NLP_SOLVER_X0)
    
    # Bounds on x
    solver.setInput(vars_lb,NLP_SOLVER_LBX)
    solver.setInput(vars_ub,NLP_SOLVER_UBX)
    
    # Bounds on g
    solver.setInput(lbg,NLP_SOLVER_LBG)
    solver.setInput(ubg,NLP_SOLVER_UBG)
    
    # NLP parameters
    solver.setInput(u0, NLP_SOLVER_P)

    return solver, X_offset, U_offset, E_offset, vars_lb, vars_ub, end_time, t_step, x0, u0, x, u, p, x_scaling, u_scaling, nk, parent_scenario, child_scenario, n_branches, n_scenarios
