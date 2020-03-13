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

    # Fixed model parameters
    M   = 1.0                          # mass of the cart               [kg]
    m   = 0.1                          # mass of the pendulum           [kg]
    l   = 0.5                          # length of lever arm            [m]
    g   = 9.81                         # grav acceleration              [m/s^2]
    uc  = 0.0005                       # coeff friction cart on track
    up  = 0.000002                     # coeff fricition pendulum on cart


    # States struct (optimization variables):
    x     = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
    v     = model.set_variable(var_type='_x', var_name='v', shape=(1,1))
    theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
    omega = model.set_variable(var_type='_x', var_name='omega', shape=(1,1))

    # Input struct (optimization variables):
    F     = model.set_variable(var_type='_u', var_name='F')

    # Fixed parameters:
    D = model.set_variable(var_type='_p', var_name='D')
    #D = 0 # Later implemend dynamic disturbance on pendulum

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    x_pendulum = model.set_expression(expr_name='x_pendulum', expr=x+l*casadi.cos(theta))

    # Expressions can also be formed without beeing explicitly added to the model.
    # The main difference is that they will not be monitored and can only be used within the current file.
    F_t = F + D
    expr1  = (-F - m*l*omega**2*casadi.sin(theta)+uc*casadi.sign(v))/(M+m)
    expr2  = l*(4/3-(m*casadi.cos(theta)**2)/(M+m)) 
    domega = ((g*casadi.sin(theta)+casadi.cos(theta)*expr1)-up*omega/(m*l))/expr2
    # Differential equations
    model.set_rhs('x', v)
    model.set_rhs('v', (-m*l*casadi.cos(theta)*domega+m*l*casadi.sin(theta)*omega**2 + F - uc*casadi.sign(v))/(M+m))
    model.set_rhs('theta', omega)
    model.set_rhs('omega', domega)

    # Build the model
    model.setup_model()


    return model
