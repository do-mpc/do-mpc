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

    R           = 8.314    			#gas constant
    T_F         = 25 + 273.15       #feed temperature
    E_a         = 8500.0     			#activation energy
    delH_R      = 950.0*1.00      			#sp reaction enthalpy
    A_tank      = 65.0       			#area heat exchanger surface jacket 65

    k_0         = 7.0*1.00      	#sp reaction rate
    k_U2        = 32.0     	#reaction parameter 1
    k_U1        = 4.0      	#reaction parameter 2
    w_WF        = .333      #mass fraction water in feed
    w_AF        = .667      #mass fraction of A in feed

    m_M_KW      = 5000.0      #mass of coolant in jacket
    fm_M_KW     = 300000.0    #coolant flow in jacket 300000;
    m_AWT_KW    = 1000.0      #mass of coolant in EHE
    fm_AWT_KW   = 100000.0    #coolant flow in EHE
    m_AWT       = 200.0       #mass of product in EHE
    fm_AWT      = 20000.0     #product flow in EHE
    m_S         = 39000.0     #mass of reactor steel

    c_pW        = 4.2      #sp heat cap coolant
    c_pS        = .47       #sp heat cap steel
    c_pF        = 3.0         #sp heat cap feed
    c_pR        = 5.0         #sp heat cap reactor contents

    k_WS        = 17280.0     #heat transfer coeff water-steel
    k_AS        = 3600.0      #heat transfer coeff monomer-steel
    k_PS        = 360.0       #heat transfer coeff product-steel

    alfa        = 5*20e4*3.6

    p_1 = 1.0

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    delH_R      = SX.sym("delH_R")
    k_0			= SX.sym("k_0")
    bias_term   = SX.sym("bias_term")

    # Define the differential states as CasADi symbols

    m_W             = SX.sym("m_W")
    m_A             = SX.sym("m_A")
    m_P		        = SX.sym("m_P")
    T_R       		= SX.sym("T_R")
    T_S  		    = SX.sym("T_S")
    Tout_M          = SX.sym("Tout_M")
    T_EK      	    = SX.sym("T_EK")
    Tout_AWT        = SX.sym("Tout_AWT")
    accum_momom		= SX.sym("accum_monom")
    T_adiab			= SX.sym("T_adiab")

    # Define the algebraic states as CasADi symbols

    # Define the control inputs as CasADi symbols
    m_dot_f    		=    SX.sym("m_dot_f")
    T_in_M   		=    SX.sym("T_in_M")
    T_in_EK       	=    SX.sym("T_in_EK")
    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    # Time constant for the hks is different if cooling or if heating
    ## --- Algebraic equations ---
    U_m    = m_P / (m_A + m_P)
    m_ges  = m_W + m_A + m_P
    k_R1   = k_0 * exp(- E_a/(R*T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
    k_R2   = k_0 * exp(- E_a/(R*T_EK))* ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
    k_K    = ((m_W / m_ges) * k_WS) + ((m_A/m_ges) * k_AS) + ((m_P/m_ges) * k_PS)

    ## --- Differential equations ---

    ddm_W   	= m_dot_f * w_WF
    ddm_A 		= (m_dot_f * w_AF) - (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) - (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
    ddm_P  		= (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT)

    ddT_R   	= 1./(c_pR * m_ges)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_K *A_tank* (T_R - T_S)) - (fm_AWT * c_pR * (T_R - T_EK)) + (delH_R * k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))))
    ddT_S   	= 1./(c_pS * m_S)     * ((k_K *A_tank* (T_R - T_S)) - (k_K *A_tank* (T_S - Tout_M)))
    ddTout_M    = 1./(c_pW * m_M_KW)  * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K *A_tank* (T_S - Tout_M)))
    ddT_EK   	= 1./(c_pR * m_AWT)   * ((fm_AWT * c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT * delH_R))
    ddTout_AWT  = 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK)))

    ddaccum_momom = m_dot_f
    ddT_adiab = delH_R/(m_ges*c_pR)*ddm_A-(ddm_A+ddm_W+ddm_P)*(m_A*delH_R/(m_ges*m_ges*c_pR))+ddT_R

    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = vertcat(m_W, m_A, m_P, T_R, T_S, Tout_M, T_EK, Tout_AWT, accum_momom, T_adiab)

    _u = vertcat(m_dot_f,T_in_M,T_in_EK)

    _xdot = vertcat(ddm_W, ddm_A, ddm_P, ddT_R, ddT_S, ddTout_M, ddT_EK, ddTout_AWT, ddaccum_momom, ddT_adiab)

    _p = vertcat(delH_R, k_0)

    _z = []


    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states
    # Initial conditions
    m_W_0 = 10000.0
    m_A_0 = 853.0*1.0  #3700.0
    m_P_0 = 26.5
    T_R_0  = 90 + 273.15
    T_S_0  = 90 + 273.15
    Tout_M_0  = 90 + 273.15
    T_EK_0 = 35 + 273.15
    Tout_AWT_0= 35 + 273.15

    accum_momom_0   = 300.0

    # This value is used here only to compute the initial condition of this state
    # This should be changed in case the real value is different
    delH_R_real = 950.0*1.00
    T_adiab_0		= m_A_0*delH_R_real/((m_W_0+m_A_0+m_P_0)*c_pR)+T_R_0

    x0   = NP.array([m_W_0, m_A_0, m_P_0, T_R_0, T_S_0, Tout_M_0, T_EK_0, Tout_AWT_0, accum_momom_0,T_adiab_0])

    # Bounds for the states and initial guess
    temp_range = 2.0
    m_W_lb          = 0;    					m_W_ub      = inf      # Kg
    m_A_lb       	= 0;    					m_A_ub      = inf      # Kg
    m_P_lb       	= 26.0;    					m_P_ub      = inf      # Kg
    T_R_lb     		= 363.15-temp_range;   		T_R_ub   	= 363.15+temp_range+10 # K
    T_S_lb 			= 298.0;    				T_S_ub 		= 400.0      # K
    Tout_M_lb       = 298.0;    				Tout_M_ub   = 400.0      # K
    T_EK_lb    		= 288.0;    				T_EK_ub    	= 400.0      # K
    Tout_AWT_lb     = 288.0;    				Tout_AWT_ub = 400.0      # K
    accum_momom_lb  = 0;						accum_momom_ub = 30000
    T_adiab_lb         =-inf;							T_adiab_ub	=  382.15 + 10 # (implemented as soft constraint)
    x_lb  = NP.array([m_W_lb, m_A_lb, m_P_lb, T_R_lb, T_S_lb, Tout_M_lb, T_EK_lb, Tout_AWT_lb, accum_momom_lb,T_adiab_lb])
    x_ub  = NP.array([m_W_ub, m_A_ub, m_P_ub, T_R_ub, T_S_ub, Tout_M_ub, T_EK_ub, Tout_AWT_ub, accum_momom_ub,T_adiab_ub])

    # Bounds for inputs
    m_dot_f_lb = 0.0;  	 m_dot_f_ub = 3.0e4;
    T_in_M_lb  = 333.15;	 T_in_M_ub  = 373.15;
    T_in_EK_lb = 333.15;   T_in_EK_ub = 373.15;

    u_lb=NP.array([m_dot_f_lb, T_in_M_lb, T_in_EK_lb])
    u_ub=NP.array([m_dot_f_ub, T_in_M_ub, T_in_EK_ub])

    # Initial guess for input (or initial condition if penalty for control movements)
    m_dot_f_0 = 0
    T_in_M_0  = 363.0
    T_in_EK_0 = 323.0

    u0   = NP.array([m_dot_f_0 , T_in_M_0, T_in_EK_0 ])

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling=NP.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10,1])
    u_scaling = NP.array([100.0, 1.0, 1.0])

    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    cons = vertcat(T_R, T_adiab)
    # Define the upper bounds of the constraint (leave it empty if not necessary)
    cons_ub = NP.array([363.15+temp_range, 382.15])
    #cons_ub = NP.array([])
    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 1
    # l1 - Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    penalty_term_cons = NP.array([1e5, 1e5])
    # Maximum violation for the soft constraints
    maximum_violation = NP.array([10, 10])

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
    lterm =  - m_P
    # Mayer term
    mterm =  - m_P
    # Penalty term for the control movements
    rterm = 0.04 * NP.array([.05,.1,.05])



    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z,'x0': x0,'x_lb': x_lb,'x_ub': x_ub, 'u0':u0, 'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'u_scaling':u_scaling, 'cons':cons,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm}

    model = core_do_mpc.model(model_dict)

    return model
