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

# imports
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
    model_type = 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters
    c0 = .4
    c1 = .2

    # State struct (hunter and prey, optimization variables)
    x_0 = model.set_variable('_x', 'x_0')
    x_1 = model.set_variable('_x', 'x_1')

    # Input struct (optimization paramters)
    inp = model.set_variable('_u', 'inp', input_type_integer=True)

    # Differential equations
    model.set_rhs('x_0', x_0 - x_0 * x_1 - c0 * x_0 * inp)
    model.set_rhs('x_1', -x_1 + x_0 * x_1 - c1 * x_1 * inp)

    # Build the model
    model.setup()

    # end of function
    return model