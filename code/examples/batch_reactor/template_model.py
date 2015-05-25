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

def template_model():
    
    """
    --------------------------------------------------------------------------
    template_model: define the non-uncertain parameters
    --------------------------------------------------------------------------
    """  
    
    mu_m	= 0.02 
    K_m	= 0.05
    K_i	= 5.0
    v_par	= 0.004
    Y_p	= 1.2

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """     
    # Define the uncertainties as CasADi symbols
    
    Y_x   = SX.sym("Y_x")
    S_in  = SX.sym("S_in")
    
    # Define the differential states as CasADi symbols
    
    X_s    = SX.sym("X_s") # Concentration A
    S_s    = SX.sym("S_s") # Concentration B
    P_s    = SX.sym("P_s") # Reactor Temprature
    V_s    = SX.sym("V_s") # Jacket Temprature
    
    # Define the algebraic states as CasADi symbols
    
    # Define the control inputs as CasADi symbols
    
    inp      = SX.sym("inp") # Vdot/V_R [h^-1]
    
    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """ 
    # Define the algebraic equations
    
    mu_S	= mu_m*S_s/(K_m+S_s+(S_s**2/K_i))
    	
    # Define the differential equations
     
    dX_s	= mu_S*X_s - inp/V_s*X_s
    dS_s	= -mu_S*X_s/Y_x - v_par*X_s/Y_p + inp/V_s*(S_in-S_s)
    dP_s	= v_par*X_s - inp/V_s*P_s 
    dV_s	= inp
    
    # Concatenate differential states, algebraic states, control inputs and right-hand-sides
    
    _x = vertcat([X_s, S_s, P_s, V_s])
    
    _u = vertcat([inp])
    
    _xdot = vertcat([dX_s, dS_s, dP_s, dV_s])
    
    _p = vertcat([Y_x, S_in])
    
    _z = []
    

    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """      
    # Initial condition for the states
    X_s_0 = 1.0
    S_s_0 = 0.5 
    P_s_0 = 0.0
    V_s_0 = 120.0 
    x0 = NP.array([X_s_0, S_s_0, P_s_0, V_s_0])
    
    # Bounds on the states. Use "inf" for unconstrained states
    X_s_lb = 0.0;			X_s_ub = 3.7
    S_s_lb = -0.01;		      S_s_ub = inf
    P_s_lb = 0.0;		      P_s_ub = 3.0
    V_s_lb = 0.0;	           V_s_ub = inf
    x_lb = NP.array([X_s_lb, S_s_lb, P_s_lb, V_s_lb])
    x_ub = NP.array([X_s_ub, S_s_ub, P_s_ub, V_s_ub])
    
    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    inp_lb = 0.0;                 inp_ub = 0.2; 	
			
    u_lb = NP.array([inp_lb])
    u_ub = NP.array([inp_ub])
    u0 = NP.array([0.03])
    
    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.array([1.0, 1.0, 1.0, 1.0])
    u_scaling = NP.array([1.0])
    
    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    cons = vertcat([])
    # Define the upper bounds of the constraint (leave it empty if not necessary)

    cons_ub = NP.array([])
    
    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 0
    # l1 - Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    penalty_term_cons = NP.array([1e4])
    # Maximum violation for the upper and lower bounds
    maximum_violation = NP.array([10])   
    
    # Define the terminal constraint (leave it empty if not necessary)
    cons_terminal = vertcat([])
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    cons_terminal_lb = NP.array([])
    cons_terminal_ub = NP.array([])   

    
    """
    --------------------------------------------------------------------------
    template_model: cost function
    --------------------------------------------------------------------------
    """         
    # Define the cost function
    # Mayer term
    mterm =  -P_s  # maximize the penicillin production
    # Lagrange term
    lterm =  0
    # Penalty term for the control movements
    rterm = 0*NP.array([0.00001])

    return (_x, _u, _xdot, _p, _z, x0, x_lb, x_ub, +
			u0, u_lb, u_ub, x_scaling, u_scaling, cons, cons_ub, +
            cons_terminal, cons_terminal_lb, cons_terminal_ub, +
            soft_constraint, penalty_term_cons, maximum_violation, +
            mterm, lterm, rterm)
