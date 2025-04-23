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
import casadi as cas
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
    K0_1 = 2.145e10      # [min^-1]
    K0_2 = 2.145e10      # [min^-1]
    E_R_1 = 9758.3       # [K]
    E_R_2 = 9758.3       # [K]
    delH_R_1 = -4200     # [kJ/kmol]
    del_H_R_2 = -11000   # [kJ/kmol]
    T_in = 387.05        # [K]
    rho = 934.2          # [kg/m^3]
    cp = 3.01            # [kJ/m^3.K]
    cp_J = 2             # [kJ/m^3.K]
    m_j = 5              # [kg]
    kA = 14.448          # [kJ/min.K]
    C_ain = 5.1          # [kmol/m^3]
    V = 0.01             # [m^3]

    # States struct (optimization variables):
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_J = model.set_variable(var_type='_x', var_name='T_J', shape=(1,1))

    # Input struct (optimization variables):
    F = model.set_variable(var_type='_u', var_name='F')
    Q_J = model.set_variable(var_type='_u', var_name='Q_J')

    # Auxiliary terms
    r_1 = K0_1 * cas.exp((-E_R_1)/((T_R)))*C_a
    r_2 = K0_2 * cas.exp((-E_R_2)/((T_R)))*C_b

    # Aux expression from auxiliary terms
    r = cas.vertcat(r_1, r_2)
    model.set_expression(expr_name='r', expr=r)

    # Differential equations
    model.set_rhs('C_a', (F/V)*(C_ain-C_a)-r_1)
    model.set_rhs('C_b', -(F/V)*C_b + r_1 - r_2)
    model.set_rhs('T_R', (F/V)*(T_in-T_R)-(kA/(rho*cp*V))*(T_R-T_J)+(1/(rho*cp))*((delH_R_1*(-r_1))+(del_H_R_2*(-r_2))))
    model.set_rhs('T_J', (1/(m_j*cp_J))*(-Q_J+kA*(T_R-T_J)))

    # Build the model
    model.setup()

    # Steady state values
    F_ss = 0.002365    # [m^3/min]
    Q_ss = 18.5583     # [kJ/min]

    C_ass = 1.6329     # [kmol/m^3]
    C_bss = 1.1101     # [kmolm^3]
    T_Rss = 398.6581   # [K]
    T_Jss = 397.3736   # [K]

    uss = np.array([[F_ss],[Q_ss]])
    xss = np.array([[C_ass],[C_bss],[T_Rss],[T_Jss]])

    # Linearize the non-linear model
    linearmodel = do_mpc.model.linearize(model, xss, uss)

    # returns linearized model
    return model,linearmodel
    
    