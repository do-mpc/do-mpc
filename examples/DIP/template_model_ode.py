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
from scipy import constants


def template_model():
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Certain parameters
    m0 = 0.6  # kg, mass of the cart
    m1 = 0.2  # kg, mass of the first rod
    m2 = 0.2  # kg, mass of the second rod
    L1 = 0.5  #m, length of the first rod
    L2 = 0.5  #m, length of the second rod

    d1 = m0 + m1 + m2
    d2 = (m1 / 2 + m2) * L1
    d3 = m2 * L2 / 2
    d4 = (m1 / 3 + m2) * L1**2
    d5 = m2 * L1 * L2 / 2
    d6 = m2 / 3 * L2**2

    f1 = (m1 / 2 + m2) * L1 * constants.g
    f2 = m2 / 2 * L2 * constants.g

    # States struct (optimization variables):
    pos = model.set_variable('_x',  'pos')
    theta = model.set_variable('_x',  'theta', (2,1))
    dpos = model.set_variable('_x',  'dpos')
    dtheta = model.set_variable('_x',  'dtheta', (2,1))

    # Input struct (optimization variables):
    u = model.set_variable('_u',  'force')

    # Expressions for kinetic and potential energy
    l1 = L1/2
    l2 = L2/2
    J1 = (m1 * l1**2) / 3
    J2 = (m2 * l2**2) / 3

    T1 = 1 / 2 * m0 * dpos**2
    T2 = 1 / 2 * m1 * (
        (dpos + l1 * dtheta[0] * cos(theta[0]))**2 +
        (l1 * dtheta[0] * sin(theta[0]))**2) + 1 / 2 * J1 * dtheta[0]**2
    T3 = 1 / 2 * m2 * (
        (dpos + L1 * dtheta[0] * cos(theta[0]) + l2 * dtheta[1] * cos(theta[1]))**2 +
        (L1 * dtheta[0] * sin(theta[0]) + l2 * dtheta[1] * sin(theta[1]))**
        2) + 1 / 2 * J2 * dtheta[0]**2

    E_kin = T1 + T2 + T3

    E_pot = m1 * constants.g * l1 * cos(
    theta[0]) + m2 * constants.g * (L1 * cos(theta[0]) +
                                l2 * cos(theta[1]))

    model.set_expression('E_kin', E_kin)
    model.set_expression('E_pot', E_pot)

    # Algebraic equations
    #matrix D
    D = blockcat([
        [d1, d2 * cos(theta[0]), d3 * cos(theta[1])],
        [d2 * cos(theta[0]), d4, d5 * cos(theta[0] - theta[1])],
        [d3 * cos(theta[1]), d5 * cos(theta[0] - theta[1]), d6]
    ])

    #matrix C
    C = blockcat([
        [0, -d2 * sin(theta[0]) * dtheta[0], -d3 * sin(theta[1]) * dtheta[1]],
        [0, 0, d5 * sin(theta[0] - theta[1]) * dtheta[1]],
        [0, -d5 * sin(theta[0] - theta[1]) * dtheta[0], 0]
    ])

    #vector H*u-G
    G = vertcat(u, f1 * sin(theta[0]), f2 * sin(theta[1]))

    #reduction of order
    sz = D.shape
    zero_mat = np.zeros((sz))
    id_mat = np.eye(sz[0])

    A = blockcat([
        [id_mat, zero_mat],
        [zero_mat, D]
    ])

    B = blockcat([
        [zero_mat, id_mat],
        [zero_mat, -C]
    ])

    F = vertcat(np.zeros((3, 1)), G)


    x = vertcat(pos, theta, dpos, dtheta)

    dx = inv(A)@(B@x+F)

    # Differential equations
    model.set_rhs('pos', dx[0])
    model.set_rhs('theta', dx[1:3])
    model.set_rhs('dpos', dx[3])
    model.set_rhs('dtheta', dx[4:6])

    # Build the model
    model.setup()

    return model
