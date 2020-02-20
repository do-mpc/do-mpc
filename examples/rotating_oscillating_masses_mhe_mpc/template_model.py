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
sys.path.append('../../')
import do_mpc


def template_model():
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Same example as shown in the Jupyter Notebooks.

    # Model variables:
    phi = model.set_variable(var_type='_x', var_name='phi', shape=(3,1))
    dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(3,1))

    # Two states for the desired (set) motor position:
    phi_m_set = model.set_variable(var_type='_u', var_name='phi_m_set', shape=(2,1))

    # Two additional states for the true motor position:
    phi_m = model.set_variable(var_type='_x', var_name='phi_m', shape=(2,1))

    # Set point for the central mass:
    phi_set = model.set_variable(var_type='_tvp', var_name='phi_set')

    # State measurements
    phi_meas = model.set_meas('phi_1_meas', phi)

    # Input measurements
    phi_m_set_meas = model.set_meas('phi_m_set_meas', phi_m_set)

    Theta_1 = model.set_variable('parameter', 'Theta_1')
    Theta_2 = model.set_variable('parameter', 'Theta_2')
    Theta_3 = model.set_variable('parameter', 'Theta_3')

    c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
    d = np.array([6.78,  8.01,  8.82])*1e-5


    model.set_rhs('phi', dphi)

    dphi_next = vertcat(
        -c[0]/Theta_1*(phi[0]-phi_m[0])-c[1]/Theta_1*(phi[0]-phi[1])-d[0]/Theta_1*dphi[0],
        -c[1]/Theta_2*(phi[1]-phi[0])-c[2]/Theta_2*(phi[1]-phi[2])-d[1]/Theta_2*dphi[1],
        -c[2]/Theta_3*(phi[2]-phi[1])-c[3]/Theta_3*(phi[2]-phi_m[1])-d[2]/Theta_3*dphi[2],
    )

    model.set_rhs('dphi', dphi_next)

    tau = 1e-2
    model.set_rhs('phi_m', 1/tau*(phi_m_set - phi_m))

    model.setup_model()

    return model
