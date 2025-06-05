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

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
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
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = 1.287e12 # K0 [h^-1]
    K0_ad = 9.043e9 # K0 [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant
    E_A_ab = 9758.3*1.00 #* R_gas# [kj/mol]
    E_A_bc = 9758.3*1.00 #* R_gas# [kj/mol]
    E_A_ad = 8560.0*1.0 #* R_gas# [kj/mol]
    H_R_ab = 4.2 # [kj/mol A]
    H_R_bc = -11.0 # [kj/mol B] Exothermic
    H_R_ad = -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
    Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 #0.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass[kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

    # States struct (optimization variables):
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

    # Input struct (optimization variables):
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

    # Fixed parameters:
    alpha = model.set_variable(var_type='_p', var_name='alpha')
    beta = model.set_variable(var_type='_p', var_name='beta')

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

    # Expressions can also be formed without being explicitly added to the model.
    # The main difference is that they will not be monitored and can only be used within the current file.
    K_1 = beta * K0_ab * exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * exp((-alpha*E_A_ad)/((T_R+273.15)))

    # Differential equations
    model.set_rhs('C_a', F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2))
    model.set_rhs('C_b', -F*C_b + K_1*C_a - K_2*C_b)
    model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
    model.set_rhs('T_K', (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k))

    # Build the model
    model.setup()

    # end of function
    return model
