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


class model_data:
    def __init__(self, model):
        self._time = np.empty((0, 1))
        self._x = np.empty((0, model.n_x))
        self._u = np.empty((0, model.n_u))
        self._z = np.empty((0, model.n_z))
        self._p = np.empty((0, model.n_p))
        self._tvp = np.empty((0, model.n_tvp))
        # TODO: n_aux not existing
        #self._aux = np.empty((0, model.n_aux))

        self.data_fields = [
            '_time',
            '_x',
            '_u',
            '_z',
            '_p',
            '_tvp'
        ]

    def export(self):
        """The export method returns a dictionary of the stored data.

        :return: Dictionary of the currently stored data.
        :rtype: dict
        """
        export_dict = {field_name: getattr(self, field_name) for field_name in self.data_fields}
        return export_dict


class optimizer_data(model_data):
    """Extension of the model_data class. All model specific fields are inherited and optimizer specific values are added.
    These include information about:
    - _cost
    - _cpu
    """
    def __init__(self, model):
        super().__init__(model)

        self._cost = np.empty((0, 1))
        self._cpu = np.empty((0, 1))

        self.data_fields.extend([
            '_cost',
            '_cpu'
        ])


class observer_data(model_data):
    def __init__(self, model):
        super().__init__(model)

        self.data_fields.extend([])


class mhe_data(observer_data):
    def __init__(self, model):
        super().__init__(model)

        self._cost = np.empty((0, 1))
        self._cpu = np.empty((0, 1))

        self.data_fields.extend([
            '_cost',
            '_cpu'
        ])
