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
import do_mpc.data

class estimator:
    def __init__(self, model):
        self.model = model

        assert model.flags['setup'] == True, 'Model for estimator was not setup. After the complete model creation call model.setup_model().'

        self._x0 = model._x(0)
        self._t0 = np.array([0])

    def set_initial_state(self, x0, reset_history=False):
        """Set the intial state of the estimator.
        Optionally resets the history. The history is empty upon creation of the estimator.

        :param x0: Initial state
        :type x0: numpy array
        :param reset_history: Resets the history of the estimator, defaults to False
        :type reset_history: bool (,optional)

        :return: None
        :rtype: None
        """
        assert x0.size == self.model._x.size, 'Intial state cannot be set because the supplied vector has the wrong size. You have {} and the model is setup for {}'.format(x0.size, self.model._x.size)
        assert isinstance(reset_history, bool), 'reset_history parameter must be of type bool. You have {}'.format(type(reset_history))
        if isinstance(x0, (np.ndarray, casadi.DM)):
            self._x0 = self.model._x(x0)
        elif isinstance(x0, structure3.DMStruct):
            self._x0 = x0
        else:
            raise Exception('x0 must be of tpye (np.ndarray, casadi.DM, structure3.DMStruct). You have: {}'.format(type(x0)))

        if reset_history:
            self.reset_history()


class state_feedback(estimator):
    def __init__(self, model):
        super().__init__(model)
        self.data = do_mpc.data.observer_data(self.model)

    def reset_history(self):
        """Reset the history of the estimator
        """
        self.data = do_mpc.data.observer_data(self.model)

class ekf(estimator):
    def __init__(self, model):
        super().__init__(model)
        self.data = do_mpc.data.observer_data(self.model)

    def reset_history(self):
        """Reset the history of the estimator
        """
        self.data = do_mpc.data.observer_data(self.model)

class mhe(estimator):
    def __init__(self, model):
        super().__init__(model)
        self.data = do_mpc.data.mhe_data(self.model)

    def reset_history(self):
        """Reset the history of the estimator
        """
        self.data = do_mpc.data.mhe_data(self.model)
