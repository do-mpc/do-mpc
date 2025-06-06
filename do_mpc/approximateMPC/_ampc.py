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

# Imports
import warnings
import torch
import numpy as np
import casadi as ca
from ._ampcsettings import ApproximateMPCSettings


# Feedforward NN
class FeedforwardNN(torch.nn.Module):
    """Feedforward Neural Network.

    .. versionadded:: >v4.6.0

    Args:
        n_in (int): Number of input neurons.
        n_out (int): Number of output neurons.
        n_hidden_layers (int): Number of hidden layers.
        n_neurons (int): Number of neurons in hidden layers.
        act_fn (str): Activation function.
        output_act_fn (str): Output activation function.
    """

    def __init__(self, n_in, n_out, n_hidden_layers, n_neurons, act_fn, output_act_fn):
        super().__init__()
        assert n_hidden_layers >= 0, "Number of hidden layers must be >= 0."
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_hidden_layers + 1
        self.n_neurons = n_neurons
        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(n_in, n_neurons))
                self.layers.append(self._get_activation_layer(act_fn))
            elif i == self.n_layers - 1:
                self.layers.append(torch.nn.Linear(n_neurons, n_out))
                if output_act_fn != "linear":
                    self.layers.append(self._get_activation_layer(output_act_fn))
            else:
                self.layers.append(torch.nn.Linear(n_neurons, n_neurons))
                self.layers.append(self._get_activation_layer(act_fn))

    def _get_activation_layer(self, act_fn):
        """Gets the pytorch activation layer.

        Args:
            act_fn (str): Activation function.

        Returns:
            torch.nn.Module: Activation layer.
        """
        if act_fn == "relu":
            return torch.nn.ReLU()
        elif act_fn == "tanh":
            return torch.nn.Tanh()
        elif act_fn == "leaky_relu":
            return torch.nn.LeakyReLU()
        elif act_fn == "sigmoid":
            return torch.nn.Sigmoid()
        else:
            raise ValueError("Activation function not implemented.")

    def forward(self, x):
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    @torch.no_grad()
    def count_params(self):
        """Count the number of parameters in the neural network.

        Returns:
            int: Number of parameters.
        """
        n_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        return n_params


# Approximate MPC
class ApproxMPC(torch.nn.Module):
    """Neural Network Approximation of the Model Predictive Controller.

    .. versionadded:: >v4.5.1

    Use this class to configure and run the ApproxMPC controller
    based on a previously configured :py:class:`do_mpc.controller.MPC` instance.

    **Configuration and setup:**

    Configuring and setting up the ApproxMPC controller involves the following steps:

    1. Configure the ApproxMPC controller with :py:class:`ApproximateMPCSettings`. The ApproxMPC instance has the attribute ``settings`` which is an instance of :py:class:`ApproximateMPCSettings`.

    2. To finalize the class configuration :py:meth:`setup` may be called. This method sets up the neural network and the MPC controller.

    **Usage:**

    The ApproxMPC controller can be used in a closed loop setting by calling the :py:meth:`make_step` method. This method takes the current state of the system and returns the control action.

    **Example:**

    ::

        # Create an MPC instance
        mpc = do_mpc.controller.MPC(...)
        mpc.setup()

        # Create an ApproxMPC instance
        ampc = do_mpc.controller.ApproxMPC(mpc)
        ampc.setup()

        # Closed loop simulation
        for k in range(N):
            x = get_current_state()
            u = ampc.make_step(x)


    Args:
        mpc (do_mpc.controller.MPC): MPC instance to be approximated by the neural network.
    """

    def __init__(self, mpc):
        # initiates torch.nn.Module
        super().__init__()

        # storage
        self._settings = ApproximateMPCSettings()
        self.mpc = mpc
        self._settings.lbx = torch.tensor(ca.DM(self.mpc._x_lb).full())
        self._settings.ubx = torch.tensor(ca.DM(self.mpc._x_ub).full())
        self._settings.lbu = torch.tensor(ca.DM(self.mpc._u_lb).full())
        self._settings.ubu = torch.tensor(ca.DM(self.mpc._u_ub).full())
        # flags
        self.flags = {
            "setup": False,
        }



    @property
    def settings(self):
        """
        All necessary parameters of the mpc formulation.

        This is a core attribute of the ApproxMPC class. It is used to set and change parameters when setting up the controller
        by accessing an instance of :py:class:`ApproximateMPCSettings`.

        Example to change settings:

        ::

            ApproxMPC.settings.n_hidden_layers = 3

        Note:
            Settings cannot be updated after calling :py:meth:`do_mpc.controller.ApproxMPC.setup`.

        For a detailed list of all available parameters see :py:class:`ApproximateMPCSettings`.
        """
        return self._settings

    @settings.setter
    def settings(self, val):
        warnings.warn("Cannot change the settings attribute")

    def setup(self):
        """Setup the ApproxMPC controller.

        Internally, this method sets the device, initializes the neural network, and sets the box constraints.

        Returns:
            None: None
        """
        assert self.flags["setup"] is False, "Setup can only be once."
        self.flags.update(
            {
                "setup": True,
            }
        )
        # sets the device
        self._set_device()

        # ampc initilisation
        if self.mpc.flags["set_rterm"]:
            self.net = FeedforwardNN(
                n_in=self.mpc.model.n_x + self.mpc.model.n_u,
                n_out=self.mpc.model.n_u,
                n_hidden_layers=self.settings.n_hidden_layers,
                n_neurons=self.settings.n_neurons,
                act_fn=self.settings.act_fn,
                output_act_fn=self.settings.output_act_fn,
            )
        else:
            self.net = FeedforwardNN(
                n_in=self.mpc.model.n_x,
                n_out=self.mpc.model.n_u,
                n_hidden_layers=self.settings.n_hidden_layers,
                n_neurons=self.settings.n_neurons,
                act_fn=self.settings.act_fn,
                output_act_fn=self.settings.output_act_fn)

        # Default settings
        self.torch_data_type = torch.float32
        self.step_return_type = "numpy"  # "torch" or "numpy"





        # storing initial guess
        self.x0 = self.mpc.x0
        self.u0 = self.mpc.u0

        print("----------------------------------")
        print(self)
        print("----------------------------------")
        assert any(torch.isinf(self.settings.lbx))==False, "There are missing lower bounds for state variables that are required for clipping and scaling."
        assert any(torch.isinf(
            self.settings.ubx))==False, "There are missing upper bounds for state variables that are required for clipping and scaling."
        assert any(torch.isinf(
            self.settings.lbu))==False, "There are missing lower bounds for input variables that are required for clipping and scaling."
        assert any(torch.isinf(
            self.settings.ubu))==False, "There are missing upper bounds for input variables that are required for clipping and scaling."

        # setup box constraints
        self.set_shift_values()

        # setup ends
        return None

    def _set_device(self):
        """Sets the device for the neural network.
        Available options for the are 'auto', 'cuda', and 'cpu'.

        Returns:
            None
        """

        # auto choose gpu if gpu is available
        if self.settings.device == "auto":
            self.settings.device = "cuda" if torch.cuda.is_available() else "cpu"

        # torch default device is set
        self.torch_device = torch.device(self.settings.device)
        torch.set_default_device(self.torch_device)

        return None

    def set_shift_values(self):
        """Shifts and scales the input and output data based on the box constraints.

        Returns:
            None
        """
        if self.mpc.flags["set_rterm"]:
            lb = torch.concatenate((self.settings.lbx.T, self.settings.lbu.T), axis=1)
            ub = torch.concatenate((self.settings.ubx.T, self.settings.ubu.T), axis=1)
        else:
            lb = self.settings.lbx.T
            ub = self.settings.ubx.T
        self.x_shift = torch.tensor(lb)
        self.x_range = torch.tensor(ub - lb)
        self.y_shift = torch.tensor(self.settings.lbu.T)
        self.y_range = torch.tensor(self.settings.ubu.T - self.settings.lbu.T)

        return None

    def forward(self, x):
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # x_scaled = self.scale_inputs(x)
        y = self.net(x)
        # y = self.rescale_outputs(y_scaled)
        return y

    def scale_inputs(self, x):
        """Scales inputs

        Args:
            x (torch.Tensor): Inputs.

        Returns:
            x_scaled (torch.Tensor): Scaled inputs.
        """
        x_scaled = (x - self.x_shift) / self.x_range
        x_scaled = x_scaled.type(self.torch_data_type)
        return x_scaled

    def rescale_outputs(self, y_scaled):
        """Rescale outputs.

        Args:
            y_scaled (torch.Tensor): Scaled outputs.

        Returns:
            y (torch.Tensor): Rescaled outputs.
        """
        y = y_scaled * self.y_range + self.y_shift
        return y

    def clip_control_actions(self, y):
        """Clip (rescaled) outputs of net to satisfy input (control) constraints.

        Args:
            y (torch.Tensor): Outputs.

        Returns:
            y (torch.Tensor): Clipped outputs.
        """
        if self.settings.lbu is not None:
            y = torch.max(y, self.settings.lbu.T)
        if self.settings.ubu is not None:
            y = torch.min(y, self.settings.ubu.T)
        if self.settings.lbu is None and self.settings.ubu is None:
            raise ValueError("No output constraints defined. Clipping not possible.")
        return y

    # Predict (Batch)
    @torch.no_grad()
    def predict(self, x_batch):
        """Predicts the output of the neural network for a batch of inputs.

        Args:
            x_batch (torch.Tensor): Batch of inputs.

        Returns:
            y_batch (torch.Tensor): Batch of outputs.
        """
        y_batch = self.net(x_batch)
        return y_batch

    # approximate MPC step method for use in closed loop
    @torch.no_grad()
    def make_step(self, x0, u_prev=None, clip_to_bounds=True):
        """Make one step with the approximate MPC.
        Args:
            x (torch.tensor): Input tensor of shape (n_in,).
            clip_to_bounds (bool, optional): Whether to clip the control actions. Defaults to True.
        Returns:
            np.array: Array of shape (n_out,).
        """
        # Check setup.
        assert self.flags['setup'] == True, 'MPC was not setup yet. Please call ApproxMPC.setup().'
        assert isinstance(x0,np.ndarray), "x0 must be a numpy array"
        assert isinstance(u_prev, (np.ndarray, type(None))), "u_prev must be a numpy array or None"

        # taking optional input if provided
        if u_prev is not None:
            self.u0 = u_prev

        if self.mpc.flags["set_rterm"]:
            x = np.concatenate((x0, ca.DM(self.u0).full()), axis=0).squeeze()
        else:
            x = x0

        # Check if inputs are tensors

        if self.mpc.flags["set_rterm"]:
            x = torch.tensor(x, dtype=self.torch_data_type).reshape(
                (-1, self.mpc.model.n_x + self.mpc.model.n_u)
            )
        else:
            x = torch.tensor(x, dtype=self.torch_data_type).reshape(
                (-1, self.mpc.model.n_x)
            )
        # forward pass
        if self.settings.scaling:
            x_scaled = self.scale_inputs(x)
            y_scaled = self.net(x_scaled)
            y = self.rescale_outputs(y_scaled)
        else:
            y = self.net(x)

        # Clip outputs to satisfy input constraints of MPC
        if clip_to_bounds:
            y = self.clip_control_actions(y)

        if self.step_return_type == "numpy":
            y = y.cpu().numpy().reshape((-1, 1))
        elif self.step_return_type == "torch":
            y = y
        else:
            raise ValueError("step_return_type must be either 'numpy' or 'torch'.")

        # storing
        self.u0 = y

        return y

    def save_to_state_dict(self, directory="approx_mpc.pth"):
        """Save the neural network to a state dictionary.

        Args:
            directory (str, optional): Directory to save the model. Defaults to "approx_mpc.pth".
        """
        torch.save(self.net.state_dict(), directory)

    def load_from_state_dict(self, directory="approx_mpc.pth"):
        """Load the neural network from a state dictionary.

        Args:
            directory (str, optional): Directory to load the model. Defaults to "approx_mpc.pth".
        """
        self.net.load_state_dict(torch.load(directory))
