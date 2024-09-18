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

rel_do_mpc_path = os.path.join('..', '..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """

    """ model and parameters adapted from "
    Cataffo, Vittorio, et al. "A nonlinear model predictive control strategy for autonomous racing 
    of scale vehicles." 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2022.")
    """
    model_type = 'continuous'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    m = 5.692  # Mass [kg]
    I_z = 0.204  # Moment of inertia around z-axis [kg*m^2]
    lf = 0.178  # Distance from CoG to front wheel [m]
    lr = 0.147  # Distance from CoG to rear wheel [m]
    D_f = 134.585  # Peak factor for front wheel [N]
    D_r = 159.919  # Peak factor for rear wheel [N]
    C_f = 0.085  # shape factor for front wheel
    C_r = 0.133  # shape factor for rear wheel
    B_f = 9.242  # stiffness factor for front wheel
    B_r = 17.716  # stiffness factor for rear wheel
    c_m1 = 20
    c_m2 = 6.92 * 1e-7
    c_m3 = 3.99
    c_m4 = 0.67

    # State struct :
    X_p = model.set_variable(var_type='_x', var_name='X_p', shape=(1, 1))
    Y_p = model.set_variable(var_type='_x', var_name='Y_p', shape=(1, 1))
    Psi = model.set_variable(var_type='_x', var_name='Psi', shape=(1, 1))  # yaw angle
    V_x = model.set_variable(var_type='_x', var_name='V_x', shape=(1, 1))
    V_y = model.set_variable(var_type='_x', var_name='V_y', shape=(1, 1))
    W = model.set_variable(var_type='_x', var_name='W', shape=(1, 1))  # yaw rate

    # Input struct (optimization variables):
    Delta = model.set_variable(var_type='_u', var_name='Delta')  # Steering angle
    d = model.set_variable(var_type='_u', var_name='d')  # Pulse Width Modulation

    # # Set expression. These can be used in the cost function, as non-linear constraints
    # # or just to monitor another output.
    Vel = model.set_expression(expr_name='Vel', expr=sqrt(V_x ** 2 + V_y ** 2))

    # Algebric Equations

    alpha_f = -atan2(W * lf + V_y, V_x) + Delta  # Front wheel slip angle
    alpha_r = atan2((W * lr - V_y), V_x)  # Rear wheel slip angle

    F_f_y = D_f * sin(C_f * atan(B_f * alpha_f))  # Front wheel force
    F_r_y = D_r * sin(C_r * atan(B_r * alpha_r))  # Rear wheel force
    F_x = (c_m1 - c_m2 * V_x) * d - c_m4 * V_x ** 2 - c_m3  # Force in x-direction

    # Differential equations
    model.set_rhs('X_p', V_x * cos(Psi) - V_y * sin(Psi))
    model.set_rhs('Y_p', V_x * sin(Psi) + V_y * cos(Psi))
    model.set_rhs('Psi', W)
    model.set_rhs('V_x', (1 / m) * (F_x - F_f_y * sin(Delta) + m * V_y * W))
    model.set_rhs('V_y', (1 / m) * (F_r_y + F_f_y * cos(Delta) - m * V_x * W))
    model.set_rhs('W', (1 / I_z) * (F_f_y * lf * cos(Delta) - lf * F_x * sin(Delta) - lr * F_r_y))

    # Build the model
    model.setup()

    return model
