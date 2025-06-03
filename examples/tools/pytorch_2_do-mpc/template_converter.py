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
import casadi as ca
import torch
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

def template_converter(nn_model):
    surrogate_model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')

    # variable setup
    d_states = {}
    d_inputs = {}
    input_layer_list = []
    d_state_list = []

    # state variables
    states = surrogate_model.set_variable(var_type='_x', var_name='states', shape=(2, 1))

    # inputs
    inputs = surrogate_model.set_variable(var_type='_u', var_name='inputs', shape=(1, 1))

    # stacked inputs and states
    input_layer = ca.vertcat(states, inputs)

    # reading the layers and the biases
    for i, layer in enumerate(nn_model):

        # linear transformations
        if isinstance(layer, torch.nn.Linear):
            # extracting weight and bias
            weight = layer.weight.cpu().detach().numpy()
            bias = layer.bias.cpu().detach().numpy()

            if i == 0:
                output_layer = ca.mtimes(weight, input_layer) + bias

            else:
                output_layer = ca.mtimes(weight, output_layer) + bias

        elif isinstance(layer, torch.nn.Tanh):
            if i == 0:
                output_layer = ca.tanh(input_layer)

            else:
                output_layer = ca.tanh(output_layer)

        else:
            raise RuntimeError('{} not supported!'.format(layer))

    # setting up rhs
    surrogate_model.set_rhs('states', output_layer)

    # setting rhs
    surrogate_model.setup()

    return surrogate_model