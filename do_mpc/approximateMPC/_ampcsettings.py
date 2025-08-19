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
from dataclasses import dataclass
import os


@dataclass
class ApproximateMPCSettings:
    """Settings for :py:class:`do_mpc.approximateMPC.ApproxMPC`.

    This class contains the mandatory settings for Approximate MPC int :py:class:`do_mpc.approximateMPC.ApproxMPC`.
    This class creates an instance of type :py:class:`ApproximateMPCSettings` and adds it to its class attributes.
    """

    n_hidden_layers: int = 3
    """Number of hidden layers"""

    n_neurons: int = 50
    """Number of neurons per hidden layer"""

    act_fn: str = "tanh"
    """Activation function used after each layer"""

    output_act_fn: str = "linear"
    """Output type"""

    device: str = "auto"
    """Type of device used. Can be `auto`, `cuda` or `cpu`"""

    scaling: bool = True
    """Decides whether the state and control variables are scaled or not"""

    lbx: list = None
    """Lower bound for the state variables"""

    ubx: list = None
    """Upper bound for the state variables"""

    lbu: list = None
    """Lower bound for the control variables"""

    ubu: list = None
    """Upper bound for the control variables"""

@dataclass
class SamplerSettings:
    """Settings for :py:class:`do_mpc.approximateMPC.Sampler`.

    This class contains the mandatory settings for Approximate MPC int :py:class:`do_mpc.approximateMPC.Sampler`.
    This class creates an instance of type :py:class:`SamplerSettings` and adds it to its class attributes.
    """

    n_samples: int = None
    """Number of samples to be generated"""

    dataset_name: str = None

    trajectory_length: int = None
    """Trajectory length for closed loop sampling"""

    # variables with default values
    closed_loop_flag: bool = False
    """Decides whether the closed loop or open loop sampling is done"""

    data_dir: "str" = os.path.join(".", "sampling")
    """Location where sampled data is stored"""

    overwrite_sampler: bool = True
    """Overwrite sampler"""


    lbx: list = None
    """Lower bound for the state variables"""

    ubx: list = None
    """Upper bound for the state variables"""

    lbu: list = None
    """Lower bound for the control variables"""

    ubu: list = None
    """Upper bound for the control variables"""

    lbp: list = None
    """Lower bound for the parameters randomly sampled for the simulator in case of closed-loop sampling of a robust MPC"""

    ubp: list = None
    """Upper bound for the parameters randomly sampled for the simulator in case of closed-loop sampling of a robust MPC"""

    def check_for_mandatory_settings(self):
        """Method to assert the necessary settings required to design :py:class:`do_mpc.approximateMPC.Sampler`"""
        if self.n_samples is None:
            raise ValueError("n_samples must be set")

        if self.dataset_name is None:
            raise ValueError("Name of your dataset must be set. Use the `name` attribute.")

        if self.closed_loop_flag is True and self.trajectory_length == None:
            raise ValueError(
                "Since closed_loop_flag is True, trajectory_length must be set"
            )

        # end of function
        return None


@dataclass
class TrainerSettings:
    """Settings for :py:class:`do_mpc.approximateMPC.Trainer`.

    This class contains the mandatory settings for Approximate MPC int :py:class:`do_mpc.approximateMPC.Trainer`.
    This class creates an instance of type :py:class:`TrainerSettings` and adds it to its class attributes.
    """

    dataset_name: str = None
    """Name of the dataset to be used for training"""

    n_epochs: int = None
    """Number of epochs for training"""

    # variables with default values
    data_dir: "str" = os.path.join(".", "sampling")
    """Location where sampled data is read from"""

    results_dir: "str" = os.path.join(".", "training")
    """Location where results are stored"""

    scheduler_flag: bool = False
    """Decides whether the scheduler adapts the learning rate"""

    val: float = 0.2
    """fill"""

    batch_size: int = 1000
    """Batch size for training"""

    shuffle: bool = True
    """Shuffle the data before training or not"""

    learning_rate: float = 1e-3
    """Default learning rate from training"""

    show_fig: bool = False
    """Display training performance after training"""

    save_fig: bool = False
    """Save training performance after training as a .png file"""

    save_history: bool = False
    """Save training performance after training as a .json file"""

    print_frequency: int = 10
    """fill"""

    def check_for_mandatory_settings(self):
        """Method to assert the necessary settings required to design :py:class:`do_mpc.approximateMPC.Trainer`"""
        if self.dataset_name is None:
            raise ValueError("The dataset name must be provided")

        if self.n_epochs is None:
            raise ValueError("A number of epochs must be set")


@dataclass
class TrainerSchedulerSettings:
    """Settings for :py:class:`do_mpc.approximateMPC.Trainer`.

    This class contains the mandatory settings for Approximate MPC int :py:class:`do_mpc.approximateMPC.Trainer`.
    This class creates an instance of type :py:class:`TrainerSchedulerSettings` and adds it to its class attributes.
    """

    mode: str = "min"
    """One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’."""

    factor: float = 0.1
    """Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1."""

    patience: float = 10
    """ The number of allowed epochs with no improvement after which the learning rate will be reduced. Default: 10."""

    threshold: float = 1e-4
    """Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4."""

    threshold_mode: str = "rel"
    """One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’."""

    cooldown: float = 2
    """Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 2."""

    min_lr: float = 1e-7
    """A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 1e-7."""

    eps: float = 1e-8
    """Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8."""
