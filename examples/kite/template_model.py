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
import do_mpc


def template_model():
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Certain parameters
    L_tether = 400.0        # Tether length [m]
    A = 300.0               # Area of the kite  [m^2]
    rho = 1.0               # [kg/m^3]
    beta = 0
    c_tilde = 0.028

    # States struct (optimization variables):
    theta = model.set_variable('_x',  'theta') # zenith angle
    phi = model.set_variable('_x',  'phi') # azimuth angle
    psi = model.set_variable('_x',  'psi') # orientation kite

    # Input struct (optimization variables):
    u_tilde = model.set_variable('_u',  'u_tilde')

    # Fixed parameters:
    E_0 = model.set_variable('_p',  'E_0')
    v_0 = model.set_variable('_p', 'v_0')

    _ = model.set_expression('E_0', E_0)
    _ = model.set_expression('v_0', v_0)

    E 		= E_0 - c_tilde * u_tilde**2
    v_a 	= v_0 * E * cos(theta)
    P_D 	= (rho * v_0**2)/2.0
    T_F		= (P_D * A * cos(theta)**2 * (E+1.0) * sqrt(E**2+1.0)) * (cos(theta) * cos(beta) + sin(theta) * sin(beta) * sin(phi))

    height_kite = L_tether * sin(theta) * cos(phi)

    model.set_expression('T_F', T_F)
    model.set_expression('height_kite', height_kite)


    # Differential equations
    dphi = -v_a / (L_tether * sin(theta)) * sin(psi)
    model.set_rhs('theta', v_a / L_tether * (cos(psi) - tan(theta)/E))
    model.set_rhs('phi', dphi)
    model.set_rhs('psi', v_a/L_tether * u_tilde + dphi *(cos(theta)))


    # Build the model
    model.setup()

    # end of function
    return model
