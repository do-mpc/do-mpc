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

import os
import sys
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

def template_model():
    # init
    model = do_mpc.model.Model('continuous', symvar_type='SX')

    # Define the states
    position = model.set_variable(var_type='_x', var_name='position', shape=(1,1))
    velocity = model.set_variable(var_type='_x', var_name='velocity', shape=(1,1))

    # Define the control inputs
    f_external = model.set_variable(var_type='_u', var_name='f_external', shape=(1,1))

    # constants
    k = 10      # spring constant
    c = 2       # damping constant
    mass = 0.1  # mass of the object

    # Define the model equations
    model.set_rhs('position', velocity)
    model.set_rhs('velocity', (-k*position - c*velocity + f_external)/mass)

    # model setup
    model.setup()

    # end
    return model