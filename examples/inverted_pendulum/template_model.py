#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2018 Sergio Lucia, Alexandru Tatulea-Codrean
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
    M   = 5.0                          # mass of the cart               [kg]
    m   = 1.0                          # mass of the pendulum           [kg]
    l   = 1.0                          # length of lever arm            [m]
    h   = 0.5                          # height of rod connection point [m]
    g   = 9.81                         # grav acceleration              [m/s^2]

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertain/varying parameters as CasADi symbols
    alpha      = SX.sym("alpha")         # can be used for implementing uncertainty

    tv_param_1 = SX.sym("tv_param_1")    # can be used to implement setpoint change

    # Define the differential states as CasADi symbols
    x 	   = SX.sym("x")               # x position of the mass
    y 	   = SX.sym("y")		          # y position of the mass
    v     = SX.sym("v")              # linear velocity of the mass
    theta = SX.sym("theta")          # angle of the metal rod
    omega = SX.sym("omega")          # angular velocity of the mass
    # Define the algebraic states as CasADi symbols
    #y 		= SX.sym("y")          # the y coordinate is considered algebraic

    # Define the control inputs as CasADi symbols
    F  	= SX.sym("F")             # control force applied to the lever

    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    # Define the differential equations
    dd = SX.sym("dd",4)
    dd[0]  = v
    dd[1]  = (1.0/(M+m-m*casadi.cos(theta)))*(m*g*casadi.sin(theta)-m*l*casadi.sin(theta)*omega**2 +F)
    dd[2] = omega
    dd[3] = (1.0/l)*(dd[1]*casadi.cos(theta)+g*casadi.sin(theta))

    # Define the algebraic equations
    dy = h+l*casadi.cos(theta) - y            # the coordinates must fulfil the constraint of the lever arm length

    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = vertcat(x,v,theta,omega)

    _z = []                                # toggle if there are no AE in your model
    #_z = vertcat(y)

    _u = vertcat(F)

    _p = vertcat(alpha)

    _xdot = vertcat(dd)

    _zdot = []                          # toggle if there are no AE in your model
    #_zdot = vertcat(dtrajectory)

    _tv_p = vertcat(tv_param_1)
    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial conditions for Differential States
    x_init  = 2.5
    v_init  = 0.0
    t_init  = 1.5
    o_init  = 0.0

    #Initial conditions for Algebraic States
    y_init  = sqrt(3.0)/2.0

    x0 = NP.array([x_init, v_init, t_init, o_init])
    z0 = NP.array([])
    # Bounds on the states. Use "inf" for unconstrained states
    x_lb    =  -10.0;       x_ub  = 10.0
    v_lb    = -10.0;       v_ub  = 10.0
    t_lb    =   -100*pi;       t_ub  = 100*pi
    o_lb    = -10.0;       o_ub  = 10.0


    x_lb = NP.array([x_lb, v_lb, t_lb, o_lb])
    x_ub = NP.array([x_ub, v_ub, t_ub, o_ub])
    z_lb = NP.array([])
    z_ub = NP.array([])
    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    F_lb    = -250.0;       F_ub = 250.00 ;     F_init = 0.0	;


    u_lb=NP.array([F_lb])
    u_ub=NP.array([F_ub])
    u0 = NP.array([F_init])

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.array([1.0, 1.0, 1.0, 1.0])
    z_scaling = NP.array([])
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
    ##-----Distance to vertical position---------------------
    lterm =  (x-tv_param_1)**2 + (theta-pi/2)**2
    # Mayer term
    mterm =  0
    # Penalty term for the control movements
    rterm = 0.0001*NP.array([1.0])



    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
     """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z,'x0': x0,'x_lb': x_lb,'x_ub': x_ub, 'u0':u0, 'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'u_scaling':u_scaling, 'cons':cons,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb,'tv_p':_tv_p, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm}

    model = core_do_mpc.model(model_dict)

    return model
