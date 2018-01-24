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
import pdb
def model(A,B,E,F, pce_terms):

    """
    --------------------------------------------------------------------------
    template_model: define the non-uncertain parameters
    --------------------------------------------------------------------------
    """

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    alpha   = SX.sym("alpha")
    beta    = SX.sym("beta")
    # Define the differential states as CasADi symbols

    SoC    = SX.sym("SoC",pce_terms) # State of Charge
    T_R    = SX.sym("T_R", pce_terms) # Room Temperature
    # Define the algebraic states as CasADi symbols

    # Define the control inputs as CasADi symbols

    P_hvac = SX.sym("P_hvac")
    P_bat  = SX.sym("P_bat")
    P_grid  = SX.sym("P_grid")

    # Define time-varying parameters that can chance at each step of the prediction and at each sampling time of the MPC controller. For example, future weather predictions

    tv_param_1 = SX.sym("tv_param_1",3) # these are the disturbances [T_ext, d_sun, d_ig]
    tv_param_2 = SX.sym("tv_param_2",2) # these are the time-varying temperature constraints

    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """

    _x = vertcat(SoC, T_R)

    _z = vertcat([])

    _u = vertcat(P_hvac, P_bat, P_grid)

    # Define the differential equations
    rhs = mtimes(A, _x) + mtimes(B, _u) + mtimes(E, tv_param_1)

    # Concatenate differential states, algebraic states, control inputs and right-hand-sides



    _xdot = vertcat(rhs)

    _zdot = vertcat([])

    _p = vertcat(alpha, beta)

    _tv_p = vertcat(tv_param_1, tv_param_2)



    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states
    SoC_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
    T_R_0 = 21.0 #[C]
    x0 = NP.array([SoC_0, T_R_0])
    # No algebraic states
    z0 = NP.array([])

    # Bounds on the states. Use "inf" for unconstrained states
    SoC_lb = 0;			SoC_ub = 100.0
    T_R_lb = 10.0;			T_R_ub = 35
    x_lb = NP.array([SoC_lb, T_R_lb])
    x_ub = NP.array([SoC_ub, T_R_ub])

    # No algebraic states
    z_lb = NP.array([])
    z_ub = NP.array([])

    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    P_hvac_lb = -1000;             P_hvac_ub = +1000.0;
    P_bat_lb = -500;          P_bat_ub = 500;
    P_grid_lb = -500;         P_grid_ub = 500;
    u_lb = NP.array([P_hvac_lb, P_bat_lb, P_grid_lb])
    u_ub = NP.array([P_hvac_ub, P_bat_ub, P_grid_ub])
    u0 = (u_lb + u_ub)/2.0

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.ones(_x.shape[0])
    z_scaling = NP.array([])
    u_scaling = NP.ones(_u.shape[0])

    # here extend the initial conditions
    # pdb.set_trace()
    x0_expanded = NP.zeros(_x.shape)
    x0_expanded[0::pce_terms,0] = x0
    x0 = x0_expanded

    # Here extend the constraint upper bounds
    x_ub_expanded =  inf * NP.ones(_x.shape)
    x_lb_expanded = -inf * NP.ones(_x.shape)

    # Here use the bounds to put constraints on the expected value (first coeff of each state)
    x_ub_expanded[0::pce_terms,0] = x_ub
    x_lb_expanded[0::pce_terms,0] = x_lb
    x_ub = x_ub_expanded
    x_lb = x_lb_expanded
    # TODO: Here formulate constraints. Two possibilities
    # 1. Use constraints on the expected value (constraints on the first coeff of each state)
    # 2. Use constraints with the variance

    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)

    # cons = vertcat([mtimes(G, _x) + mtimes(H, _u) + mtimes(I, _tv_param_1), -(mtimes(G, _x) + mtimes(H, _u) + mtimes(I, _tv_param_1))])
    # # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    # cons_ub = NP.array([m_ub, -m_lb])

    cons = vertcat()
    cons_ub = NP.array([])
    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 0
    # Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    penalty_term_cons = NP.array([])
    # Maximum violation for the constraints
    maximum_violation = NP.array([0])

    # Define the terminal constraint (leave it empty if not necessary)
    cons_terminal = vertcat()
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
    lterm =  P_grid + 0 * (SoC[0] - 60)**2
    #lterm =  - C_b
    # Mayer term
    mterm =  P_grid + 0 * (SoC[0] - 60)**2
    #mterm =  - C_b
    # Penalty term for the control movements
    rterm = NP.array([0.0, 0.0,0.0])



    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z, 'aes': _zdot,'x0': x0, 'z0':z0, 'x_lb': x_lb,'x_ub': x_ub, 'z_lb': z_lb,'z_ub': z_ub, 'u0':u0,
    'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'z_scaling':z_scaling, 'u_scaling':u_scaling, 'cons':cons, 'tv_p':_tv_p,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm}

    model = core_do_mpc.model(model_dict)

    return model
