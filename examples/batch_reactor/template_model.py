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
def model():

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

    _x = vertcat(X_s, S_s, P_s, V_s)

    _u = vertcat(inp)

    _xdot = vertcat(dX_s, dS_s, dP_s, dV_s)

    _p = vertcat(Y_x, S_in)

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
    # Lagrange term
    lterm =  -P_s
    #lterm =  - C_b
    # Mayer term
    mterm =  -P_s
    #mterm =  - C_b
    # Penalty term for the control movements
    rterm = NP.array([0.0])



    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z,'x0': x0,'x_lb': x_lb,'x_ub': x_ub, 'u0':u0, 'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'u_scaling':u_scaling, 'cons':cons,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm}

    model = core_do_mpc.model(model_dict)

    return model
