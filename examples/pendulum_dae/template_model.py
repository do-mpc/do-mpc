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
    m   = 12.0                              # mass of the pendulum    [kg]
    l   = 1.0                               # length of lever arm     [m]
    x00 = 0.0                               # vertical position coord [m]
    y00 = 1.0
    g   = 9.81                              # grav acceleration   [m/s^2]

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols
    alpha   = SX.sym("alpha")               # can be used for implementing uncertainty

    # Define the differential states as CasADi symbols

    x 		= SX.sym("x")              # x coordinate of the mass
    #y 		= SX.sym("y")		    # y coordinate of the mass
    vx           = SX.sym("vx")             # velocity on the Ox axis
    vy           = SX.sym("vy")             # velocity on the Oy axis

    # Define the algebraic states as CasADi symbols
    y 		= SX.sym("y")             # the y coordinate is considered algebraic
    # Define the control inputs as CasADi symbols
    F  	= SX.sym("F")                   # control force applied to the lever

    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    ##-----Distance to vertical position---------------------
    d = sqrt((x-x00)**2+(y-y00)**2)


    # Define the differential equations
    dx    = vx
    dvx   = - (F/(m*l))*x
    dvy   = g - (F/(m*l))*y

    # Define the algebraic equations
    dtrajectory = x**2+y**2-l**2            # the coordinates must fulfill the constraint of the lever arm length

    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = vertcat(x,vx,vy)

    #_z = []                                # toggle if there are no AE in your model
    _z = vertcat(y)

    _u = vertcat(F)

    _p = vertcat(alpha)

    _xdot = vertcat(dx, dvx, dvy)

    #_zdot = vertcat([])                    # toggle if there are no AE in your model
    _zdot = vertcat(dtrajectory)

    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial conditions for Differential States
    x_init  = 0.5
    vx_init = 0.0
    vy_init = 0.0

    #Initial conditions for Algebraic States
    y_init  = sqrt(3.0)/2.0

    x0 = NP.array([x_init, vx_init, vy_init])
    z0 = NP.array([y_init])
    # Bounds on the states. Use "inf" for unconstrained states
    x_lb    =  -10.0;                                 x_ub   = 10.0;
    y_lb    =    0.0;					  	  y_ub   = 10.0
    vx_lb   = -100.0;					        vx_ub   = 100.0
    vy_lb   = -100.0;			                   vy_ub  = 100.0


    x_lb = NP.array([x_lb, vx_lb, vy_lb])
    x_ub = NP.array([x_ub, vx_ub, vy_ub])
    z_lb = NP.array([y_lb])
    z_ub = NP.array([y_ub])
    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    F_lb    = -5000.0;		                        F_ub = 5000.00 ;		         F_init = 0.0	;


    u_lb=NP.array([F_lb])
    u_ub=NP.array([F_ub])
    u0 = NP.array([F_init])

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.array([1.0, 1.0, 1.0])
    z_scaling = NP.array([1.0])
    u_scaling = NP.array([1.0])

    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    cons = vertcat([])
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
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
    lterm =  d**2
    # Mayer term
    mterm =  0

    # Penalty term for the control movements
    rterm = 0.000001*NP.array([1.0])



    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z, 'aes': _zdot,'x0': x0, 'z0':z0, 'x_lb': x_lb,'x_ub': x_ub, 'z_lb': z_lb,'z_ub': z_ub, 'u0':u0, 'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'z_scaling':z_scaling, 'u_scaling':u_scaling, 'cons':cons,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm}

    model = core_do_mpc.model(model_dict)

    return model
