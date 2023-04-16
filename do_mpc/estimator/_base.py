
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
import do_mpc
from typing import Union

class Estimator(do_mpc.model.IteratedVariables):
    """The Estimator base class. Used for :py:class:`StateFeedback`, :py:class:`EKF` and :py:class:`MHE`.
    This class cannot be used independently.

    Note:
       The methods :py:func:`Estimator.set_initial_state` and :py:func:`Estimator.reset_history`
       are overwritten when using the :py:class:`MHE` by the methods defined in :py:class:`do_mpc.optimizer.Optimizer`.

    Args:
        model: model from class :py:class:`do_mpc.model`
    """
    def __init__(self, model:Union[do_mpc.model.Model,do_mpc.model.LinearModel]):
        self.model = model
        do_mpc.model.IteratedVariables.__init__(self)

        assert model.flags['setup'] == True, 'Model for estimator was not setup. After the complete model creation call model.setup().'

        self.data = do_mpc.data.Data(model)
        self.data.dtype = 'Estimator'


    def reset_history(self)->None:
        """Reset the history of the estimator
        """
        self.data.init_storage()


class StateFeedback(Estimator):
    """Simple state-feedback "estimator".
    The main method :py:func:`StateFeedback.make_step` simply returns the input.
    Why do you even bother to use this class?
    """
    def __init__(self, model):
        super().__init__(model)

    def make_step(self, y0:np.ndarray)->np.ndarray:
        """Returns the measurement.
        
        Args:
            y0: measurment

        Returns:
            Return the measurement ``y0``.
        """
        return y0