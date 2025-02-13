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

#import data class decorator
from dataclasses import dataclass
import os
@dataclass
class ApproximateMPCSettings:
    """Settings for :py:class:`do_mpc.approximateMPC`.

    This class contains the mandaory settings for Approximate MPC int :py:class:`do_mpc.approximateMPC`.
    This class creates an instance of type :py:class:`ApproximateMPCSettings` and adds it to its class attributes.
    """
    n_hidden_layers: int = 3
    """Number of hidden layers"""

    n_neurons: int = 50
    """Number of neurons per hidden layer"""

    act_fn: str = 'relu'
    """Activation function used after each layer"""

    output_act_fn: str = 'linear'
    """Output type"""

    device: str = 'auto'
    """Type of device used. Can be auto, cuda or cpu"""

@dataclass
class SamplerSettings:
    """Settings for :py:class:`do_mpc.approximateMPC`.

    This class contains the mandaory settings for Approximate MPC int :py:class:`do_mpc.approximateMPC`.
    This class creates an instance of type :py:class:`ApproximateMPCSettings` and adds it to its class attributes.
    """

    n_samples: int = None
    """Number of samples to be generated"""

    trajectory_length: int = None
    """Trajectory length for closed loop sampling"""

    # variables with default values
    closed_loop_flag: bool = False
    """Decides whether the closed loop or open loop sampling is done"""

    data_dir: 'str' = os.path.join('.', 'sampling')
    """Location where sampled data is stored"""

    overwrite_sampler: bool = True
    """Overwrite sampler"""

    # def init for simulator settings

    def check_for_mandatory_settings(self):
        """Method to assert the necessary settings required to design :py:class:`do_mpc.controller`
        """
        if self.n_samples is None:
            raise ValueError("n_samples must be set")

        if self.closed_loop_flag is True and self.trajectory_length == None:
            raise ValueError("Since closed_loop_flag is True, trajectory_length must be set")

        # end of function
        return  None

@dataclass
class TrainerSettings:
    """Settings for :py:class:`do_mpc.approximateMPC`.

    This class contains the mandaory settings for Approximate MPC int :py:class:`do_mpc.approximateMPC`.
    This class creates an instance of type :py:class:`ApproximateMPCSettings` and adds it to its class attributes.
    """

    n_samples: int = None
    """Number of samples to be trained"""

    n_epochs: int = None
    """Number of epochs for training"""

    # variables with default values
    data_dir: 'str' = os.path.join('.', 'sampling')
    """Location where sampled data is stored"""

    results_dir: 'str' = os.path.join('.', 'training')

    scheduler_flag: bool = False
    """Decides whether the scheduler stops training after adequately completing training"""

    val: float = 0.2
    """fill"""

    batch_size: int = 1000
    """fill"""

    shuffle: bool =True
    """fill"""

    learning_rate: float = 1e-3
    """fill"""

    show_fig: bool = True
    """fill"""

    save_fig: bool = True
    """fill"""

    save_history: bool = True
    """fill"""

    print_frequency: int = 10
    """fill"""

    def check_for_mandatory_settings(self):
        """Method to assert the necessary settings required to design :py:class:`do_mpc.controller`
        """
        if self.n_samples is None:
            raise ValueError("n_samples must be set")

        if self.n_epochs is None:
            raise ValueError("n_epochs must be set")

@dataclass
class TrainerSchedulerSettings:
    """Settings for :py:class:`do_mpc.approximateMPC`.

    This class contains the mandaory settings for Approximate MPC int :py:class:`do_mpc.approximateMPC`.
    This class creates an instance of type :py:class:`ApproximateMPCSettings` and adds it to its class attributes.
    """

    mode: str = 'min'
    """0"""

    factor: float = 0.1
    """1"""

    patience: float = 10
    """2"""

    threshold: float = 1e-5
    """3"""

    threshold_mode: str = 'rel'
    """4"""

    cooldown: float = 2
    """5"""

    min_lr: float = 1e-9
    """6"""

    eps: float = 0
    """7"""