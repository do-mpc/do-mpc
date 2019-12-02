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

class state_feedback(estimator):
    def __init__(self, model):
        super().__init__(model)
        self.data = do_mpc.data.observer_data(model)

class ekf(estimator):
    def __init__(self, model):
        super().__init__(model)
        self.data = do_mpc.data.observer_data(model)

class mhe(estimator):
    def __init__(self, model):
        super().__init__(model)
        self.data = do_mpc.data.mhe_data(model)
