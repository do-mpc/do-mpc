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

    # Possibly uncertain parameters (some will be overwritten) and constants
    Vs = 230.0 		# Bus Voltage [V]

    L_eq = 19.0e-6 # Equivalent inductance [H] Uncertainty: 50 %
    R_eq = 2.9   # Equivalent resistance [Ohm] Uncertainty: 50 %
    C_res = 1440e-9 # Uncertainty: 20 %




    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    alpha   = SX.sym("alpha")
    beta    = SX.sym("beta")
    # Define the differential states as CasADi symbols

    i_0             = SX.sym("i_0")
    v_C             = SX.sym("v_C")
    my_time         = SX.sym("my_time")

    # Define the control inputs as CasADi symbols

    duty    		= SX.sym("duty")
    F				= SX.sym("F")

    # Define time-varying parameters that can chance at each step of the prediction and at each sampling time of the MPC controller. For example, future weather predictions

    tv_param_1 = SX.sym("tv_param_1")
    tv_param_2 = SX.sym("tv_param_2")
    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    ## --- Algebraic equations ---
    # Half-bridge voltage using H harmonics
    # H = 50
    # v_0 = 0
    # for index_harmonic in range(H):
    # 	i_H = index_harmonic + 1
    # 	v_0_ch = Vs/(i_H * pi) * sin(2 * pi * i_H * duty)
    # 	v_0_sh = Vs/(i_H * pi) * (1 - cos(2 * pi * i_H * duty))
    # 	v_0 += v_0_ch * cos(i_H * F*1000*(2*pi) * my_time/10000) + v_0_sh * sin(i_H * F*1000*(2*pi) * my_time/10000)
    #

    # v_0 += Vs * duty
    # Bounds on the frequency
    F_min = 1.0/(2.0*pi*sqrt(L_eq*C_res))/1000.0   #  [Hz]
    F_max = 100.0e3 /1000.0						#  [Hz]
    R_eq = R_eq * alpha
    L_eq = L_eq * beta
    # v_switch = if_else(mod(my_time/10000,1/(F*1000)) > duty/(F*1000), 0, 1)
    # v_switch = if_else(my_time > duty/F, 0, 1)
    v_switch = -0.5*(tanh(50*(my_time*F - duty)) - 1.0)
    # v_switch = -0.5*(tanh(1e7*(mod(my_time/10000.0,1.0/(F*1000)) - duty/(F*1000))) - 1.0)

    v_switch = tv_param_2
    v_0 = Vs * v_switch

    power = v_0 * i_0
    ## --- Differential equations ---

    ddi_0 = (tv_param_2 * duty/(F*1000) + (1-tv_param_2) * (1-duty)/(F*1000))  * 1.0 / L_eq * (v_0 - R_eq*i_0 - v_C)
    ddv_C = (tv_param_2 * duty/(F*1000) + (1-tv_param_2) * (1-duty)/(F*1000)) * 1.0 / C_res * (i_0)
    ddmy_time = tv_param_2 * duty/(F) + (1-tv_param_2) * (1-duty)/(F)

    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = vertcat(i_0, v_C)

    _z = vertcat([])

    _u = vertcat(duty, F)

    _xdot = vertcat(ddi_0, ddv_C)

    _zdot = vertcat([])

    _p = vertcat(alpha, beta)

    _tv_p = vertcat(tv_param_1, tv_param_2)

    _other = vertcat(v_0, power)

    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial conditions
    i_0_0 = 0
    v_C_0	= 0
    my_time_0 = 0
    # Compose initial state
    x0  = NP.array([i_0_0, v_C_0])
    # No algebraic states
    z0 = NP.array([])

    # Bounds on the states. Use "inf" for unconstrained states
    i_0_lb	= -200.0;				i_0_ub = +200.0
    v_C_lb		= -1000.0;				v_C_ub	 = +1000.0
    my_time_lb	= -0.0;						my_time_ub = 100

    x_lb  = NP.array([i_0_lb, v_C_lb])*100
    x_ub  = NP.array([i_0_ub, v_C_ub])*100

    # No algebraic states
    z_lb = NP.array([])
    z_ub = NP.array([])

    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    F_lb = F_min;  	 F_ub = F_max;
    # F_lb = 50.0;  	 F_ub = 50.0;
    duty_lb = 0.2; 	 duty_ub = 0.8;


    u_lb = NP.array([duty_lb, F_lb])
    u_ub = NP.array([duty_ub, F_ub])

    u0 = (u_lb + u_ub) / 2.0 #NP.array([duty_lb, F_lb])

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.array([1.0, 1.0])
    z_scaling = NP.array([])
    u_scaling = NP.array([1.0, 1.0])

    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    cons = vertcat()
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    cons_ub = NP.array([])

    # Define the terminal constraint (leave it empty if not necessary)
    # cons_zvs = vertcat(-power)
    cons_zvs = vertcat()
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    # cons_zvs_ub = NP.array([50])
    cons_zvs_ub = NP.array([])

    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 0
    # Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    penalty_term_cons = NP.array([])
    # Maximum violation for the constraints
    maximum_violation = NP.array([])

    # Define the terminal constraint (leave it empty if not necessary)
    cons_terminal = vertcat(i_0)
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    cons_terminal_lb = NP.array([-inf])
    cons_terminal_ub = NP.array([0])



    """
    --------------------------------------------------------------------------
    template_model: cost function
    --------------------------------------------------------------------------
    """
    # Define the cost function
    # Lagrange term
    lterm =  power
    # Mayer term
    # In this case mterm is the cost for any other goal different from power tracking
    # mterm =  (F/F_min)**2
    mterm =  1*0.0005*(F/F_min-1)**2
    # mterm =  0.00001*(F-F_min)**2
    # mterm =  0.001*F
    # Penalty term for the control movements
    rterm = NP.array([1., 0.1])*0.0
    # rterm = NP.array([1., 0.02*1/F_max])*1



    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z, 'aes': _zdot,'x0': x0, 'z0':z0, 'x_lb': x_lb,'x_ub': x_ub, 'z_lb': z_lb,'z_ub': z_ub, 'u0':u0,
    'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'z_scaling':z_scaling, 'u_scaling':u_scaling, 'cons':cons, 'tv_p':_tv_p, 'other':_other,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm,
    'cons_zvs':cons_zvs, 'cons_zvs_ub':cons_zvs_ub}

    model = core_do_mpc.model(model_dict)

    return model
