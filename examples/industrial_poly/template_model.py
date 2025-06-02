#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
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

from casadi import *
from casadi.tools import *
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters
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

    p_1         = 1.0

    # States struct (optimization variables):
    m_W =         model.set_variable('_x', 'm_W')
    m_A =         model.set_variable('_x', 'm_A')
    m_P =         model.set_variable('_x', 'm_P')
    T_R =         model.set_variable('_x', 'T_R')
    T_S =         model.set_variable('_x', 'T_S')
    Tout_M =      model.set_variable('_x', 'Tout_M')
    T_EK =        model.set_variable('_x', 'T_EK')
    Tout_AWT =    model.set_variable('_x', 'Tout_AWT')
    accum_monom = model.set_variable('_x', 'accum_monom')
    T_adiab =     model.set_variable('_x', 'T_adiab')

    # Input struct (optimization variables):
    m_dot_f = model.set_variable('_u', 'm_dot_f')
    T_in_M =  model.set_variable('_u', 'T_in_M')
    T_in_EK = model.set_variable('_u', 'T_in_EK')

    # Fixed parameters:
    delH_R = model.set_variable('_p', 'delH_R')
    k_0 =    model.set_variable('_p', 'k_0')


    # algebraic equations
    U_m    = m_P / (m_A + m_P)
    m_ges  = m_W + m_A + m_P
    k_R1   = k_0 * exp(- E_a/(R*T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
    k_R2   = k_0 * exp(- E_a/(R*T_EK))* ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
    k_K    = ((m_W / m_ges) * k_WS) + ((m_A/m_ges) * k_AS) + ((m_P/m_ges) * k_PS)

    # Differential equations
    dot_m_W = m_dot_f * w_WF
    model.set_rhs('m_W', dot_m_W)
    dot_m_A = (m_dot_f * w_AF) - (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) - (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
    model.set_rhs('m_A', dot_m_A)
    dot_m_P = (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
    model.set_rhs('m_P', dot_m_P)

    dot_T_R = 1./(c_pR * m_ges)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_K *A_tank* (T_R - T_S)) - (fm_AWT * c_pR * (T_R - T_EK)) + (delH_R * k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))))
    model.set_rhs('T_R', dot_T_R)
    model.set_rhs('T_S', 1./(c_pS * m_S)     * ((k_K *A_tank* (T_R - T_S)) - (k_K *A_tank* (T_S - Tout_M))))
    model.set_rhs('Tout_M', 1./(c_pW * m_M_KW)  * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K *A_tank* (T_S - Tout_M))))
    model.set_rhs('T_EK', 1./(c_pR * m_AWT)   * ((fm_AWT * c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT * delH_R)))
    model.set_rhs('Tout_AWT', 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK))))
    model.set_rhs('accum_monom', m_dot_f)
    model.set_rhs('T_adiab', delH_R/(m_ges*c_pR)*dot_m_A-(dot_m_A+dot_m_W+dot_m_P)*(m_A*delH_R/(m_ges*m_ges*c_pR))+dot_T_R)

    # Build the model
    model.setup()


    return model
