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


# This file describes the model and optimal control problem of an emulated lab
# reactor, taken from the following publication. Plese cite it if you use this model:

# S. Lucia. Robust Multi-stage Nonlinear Model Predictive Control. Shaker Verlag, 2015.

def template_model():

    """
    --------------------------------------------------------------------------
    template_model: define the non-uncertain parameters
    --------------------------------------------------------------------------
    """

    alpha  = 0.1237*1.2# real is 1.2	#kW/m2*K
    rho    = 1000.0		#kg/m3
    Cp	   = 4.2		# kJ/kg*K
    r      = 0.092	# m - the radius of the reactor  ***** New measurement (old measurement was 0.086)
    pi     = 3.1415		# ****
    MJ     = 2.22		# kg
    dMJin  = 11.0/60.0*0.5  #0.034		# kg/sec   *****   Maybe 11/60
    CBin   = 4000.0*1.0	# mol/kg
    Tin    = 273.15 +27# K
    T_desired = 50.0

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    k_   = SX.sym("k_")
    delH  = SX.sym("delH")

    # Define the differential states as CasADi symbols

    V_R     = SX.sym("V_R")		# m3
    Ca      = SX.sym("Ca")		# mol/m3
    Cb		= SX.sym("Cb")		# mol/m3
    Cc		= SX.sym("Cc")		# mol/m3
    T_R		= SX.sym("T_R")		# K
    T_J		= SX.sym("T_J")		# K
    T_J_in	= SX.sym("T_J_in")	# K
    i_feed  = SX.sym("i_feed")

    # Define the algebraic states as CasADi symbols

    # Define the control inputs as CasADi symbols
    u1 	= SX.sym("u1")		# V_R_in
    u2	= SX.sym("u2")		# T_J_in
    u3  = SX.sym("u3")		# reaction rate - fake heating source
    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    # Time constant for the hks is different if cooling or if heating
    tau_heat = 90
    tau_cool = 990.0
    tau_hks = if_else(u2-T_J_in>0, tau_heat, tau_cool)
    tau_hks = tau_cool
    Area = pi*r**2+2*V_R/r
    ## --- Differential equations ---
    dV_R     = u1/1000.0
    dCa      = -Ca*u1/1000.0/V_R - k_*Ca*Cb
    dCb      = -Cb*u1/1000.0/V_R + CBin*u1/1000.0/V_R - k_*Ca*Cb
    dCc      = -Cc*u1/1000.0/V_R + k_*Ca*Cb
    dT_R     = (u1)/1000.0/V_R*(Tin-T_R)-alpha*Area*(T_R-T_J)/(rho*V_R*Cp)-k_*Ca*Cb*delH/(rho*Cp)#-(u1/V_R + (alpha*(pi*r**2+2*V_R/r)/(rho*Cp*V_R)))*T_R + (alpha*(pi*r**2+2*V_R/r)*T_J)/(rho*Cp*V_R) + Tin*u1/V_R-k_*Ca*Cb*delH/(rho*Cp)
    dT_J     = alpha*Area*(T_R-T_J)/(MJ*Cp) + dMJin/(MJ/rho)*(T_J_in-T_J)# (((dMJin*Cp+alpha*(pi*r**2+2*V_R/r))*T_J)/(MJ*Cp)) + dMJin*T_J_in/MJ
    dT_J_in	 = (u2-T_J_in)/tau_hks#1015.0
    di_feed = u1/1000.0

    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = vertcat([V_R, Ca, Cb, Cc, T_R, T_J, T_J_in, i_feed])

    _u = vertcat([u1, u2])

    _xdot = vertcat([dV_R, dCa, dCb, dCc, dT_R, dT_J, dT_J_in, di_feed])

    _p = vertcat([k_, delH])

    _z = []


    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states
    V_R_0    = 0.00352
    Ca_0     = 2000.0
    Cb_0     = 0.0
    Cc_0     = 0.0
    T_R_0    = T_desired + 273.15#325.0
    T_J_0	 = 50 + 273.15
    T_J_in_0 = 50 + 273.15
    i_feed_0 = 0
    x0   = NP.array([V_R_0, Ca_0, Cb_0, Cc_0, T_R_0, T_J_0, T_J_in_0, i_feed_0])

    V_R_lb          = 1e-3;    			V_R_ub     = 7.0e-2#0.005250     # m3
    Ca_lb       	= -0.01;    			Ca_ub      = 2010.0     # mol/l
    Cb_lb       	= -0.01;				Cb_ub      = 2000.0     # mol/l
    Cc_lb       	= -0.01;				Cc_ub      = 4000.0     # mol/l
    T_R_lb			= T_desired +273.15-20;		T_R_ub	   = T_desired + 273.15+20   # K
    T_J_lb			= 273.15;			T_J_ub	   = 100 + 273.15	# K
    T_J_in_lb       = 273.15;			T_J_in_ub  = 100 + 273.15   # K
    i_feed_lb = 0; i_feed_ub = 1.0
    x_lb  = NP.array([V_R_lb, Ca_lb, Cb_lb, Cc_lb, T_R_lb, T_J_lb, T_J_in_lb, i_feed_lb])
    x_ub  = NP.array([V_R_ub, Ca_ub, Cb_ub, Cc_ub, T_R_ub, T_J_ub, T_J_in_ub, i_feed_ub])

    # Bounds for inputs
    u1_lb		= 0.000000;		u1_ub	= 0.000009*1000.0
    u2_lb		= 30 +273.15;			u2_ub   = 80 +273.15
    u3_lb		= 1.0;				u3_ub	= 1.0

    u_lb  = NP.array([u1_lb, u2_lb])
    u_ub  = NP.array([u1_ub, u2_ub])
    u1_0 = 0.000005*1000.0
    u2_0 = 30 +273.15
    u3_0 = 1.0

    u0   = NP.array([u1_0, u2_0])

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.array([1.0*1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    u_scaling = NP.array([1.0, 1.0])

    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    cons = vertcat([T_R, -T_R, i_feed*CBin])
    # Define the upper bounds of the constraint (leave it empty if not necessary)

    cons_ub = NP.array([52.0+273.15, - (48.0+273.15), Ca_0 * x0[0]])
    #cons_ub = NP.array([])
    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 1
    # l1 - Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    penalty_term_cons = NP.array([1e5, 1e5, 0])
    # Maximum violation for the upper and lower bounds
    maximum_violation = NP.array([10,10, 0])

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
    mterm =  - 10.0 * V_R*Cc   # maximize the producion of C
    # Lagrange term
    lterm =  0.0
    # Penalty term for the control movements
    rterm = NP.array([5e6,0.01])

    return (_x, _u, _xdot, _p, _z, x0, x_lb, x_ub, u0, u_lb, u_ub, x_scaling, u_scaling, cons, cons_ub, cons_terminal, cons_terminal_lb, cons_terminal_ub, soft_constraint, penalty_term_cons, maximum_violation, mterm, lterm, rterm)
