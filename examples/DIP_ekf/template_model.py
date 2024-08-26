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


def template_model(obstacles, symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters
    m0 = 0.6  # kg, mass of the cart
    m1 = 0.2  # kg, mass of the first rod
    m2 = 0.2  # kg, mass of the second rod
    L1 = 0.5  #m, length of the first rod
    L2 = 0.5  #m, length of the second rod
    l1 = L1/2
    l2 = L2/2
    J1 = (m1 * l1**2) / 3   # Inertia
    J2 = (m2 * l2**2) / 3   # Inertia

    m1 = model.set_variable('_p', 'm1')
    m2 = model.set_variable('_p', 'm2')

    g = 9.80665 # m/s^2, gravity

    h1 = m0 + m1 + m2
    h2 = m1*l1 + m2*L1
    h3 = m2*l2
    h4 = m1*l1**2 + m2*L1**2 + J1
    h5 = m2*l2*L1
    h6 = m2*l2**2 + J2
    h7 = (m1*l1 + m2*L1) * g
    h8 = m2*l2*g

    # Setpoint x:
    pos_set = model.set_variable('_tvp', 'pos_set')


    # States struct (optimization variables):
    pos = model.set_variable('_x',  'pos')
    theta = model.set_variable('_x',  'theta', (2,1))
    dpos = model.set_variable('_x',  'dpos')
    dtheta = model.set_variable('_x',  'dtheta', (2,1))
    # Algebraic states:
    ddpos = model.set_variable('_z', 'ddpos')
    ddtheta = model.set_variable('_z', 'ddtheta', (2,1))

    # Input struct (optimization variables):
    u = model.set_variable('_u',  'force')

    # Differential equations
    model.set_rhs('pos', dpos)
    model.set_rhs('theta', dtheta)
    model.set_rhs('dpos', ddpos)
    model.set_rhs('dtheta', ddtheta)

    # Euler Lagrange equations for the DIP system (in the form f(x,u,z) = 0)
    euler_lagrange = vertcat(
        # 1
        h1*ddpos+h2*ddtheta[0]*cos(theta[0])+h3*ddtheta[1]*cos(theta[1])
        - (h2*dtheta[0]**2*sin(theta[0]) + h3*dtheta[1]**2*sin(theta[1]) + u),
        # 2
        h2*cos(theta[0])*ddpos + h4*ddtheta[0] + h5*cos(theta[0]-theta[1])*ddtheta[1]
        - (h7*sin(theta[0]) - h5*dtheta[1]**2*sin(theta[0]-theta[1])),
        # 3
        h3*cos(theta[1])*ddpos + h5*cos(theta[0]-theta[1])*ddtheta[0] + h6*ddtheta[1]
        - (h5*dtheta[0]**2*sin(theta[0]-theta[1]) + h8*sin(theta[1]))
    )

    model.set_alg('euler_lagrange', euler_lagrange)

    # Expressions for kinetic and potential energy
    E_kin_cart = 1 / 2 * m0 * dpos**2
    E_kin_p1 = 1 / 2 * m1 * (
        (dpos + l1 * dtheta[0] * cos(theta[0]))**2 +
        (l1 * dtheta[0] * sin(theta[0]))**2) + 1 / 2 * J1 * dtheta[0]**2
    E_kin_p2 = 1 / 2 * m2 * (
        (dpos + L1 * dtheta[0] * cos(theta[0]) + l2 * dtheta[1] * cos(theta[1]))**2 +
        (L1 * dtheta[0] * sin(theta[0]) + l2 * dtheta[1] * sin(theta[1]))**
        2) + 1 / 2 * J2 * dtheta[0]**2

    E_kin = E_kin_cart + E_kin_p1 + E_kin_p2

    E_pot = m1 * g * l1 * cos(
    theta[0]) + m2 * g * (L1 * cos(theta[0]) +
                                l2 * cos(theta[1]))

    model.set_expression('E_kin', E_kin)
    model.set_expression('E_pot', E_pot)

    # Calculations to avoid obstacles:

    # Coordinates of the nodes:
    node0_x = model.x['pos']
    node0_y = np.array([0])

    node1_x = node0_x+L1*sin(model.x['theta',0])
    node1_y = node0_y+L1*cos(model.x['theta',0])

    node2_x = node1_x+L2*sin(model.x['theta',1])
    node2_y = node1_y+L2*cos(model.x['theta',1])

    obstacle_distance = []

    for obs in obstacles:
        d0 = sqrt((node0_x-obs['x'])**2+(node0_y-obs['y'])**2)-obs['r']*1.05
        d1 = sqrt((node1_x-obs['x'])**2+(node1_y-obs['y'])**2)-obs['r']*1.05
        d2 = sqrt((node2_x-obs['x'])**2+(node2_y-obs['y'])**2)-obs['r']*1.05
        obstacle_distance.extend([d0, d1, d2])


    model.set_expression('obstacle_distance',vertcat(*obstacle_distance))
    model.set_expression('tvp', pos_set)


    # Build the model
    model.setup()

    return model
