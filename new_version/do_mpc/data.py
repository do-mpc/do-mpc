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

    def update(self, **kwargs):
        """Update value(s) of the data structure with key word arguments.
        These key word arguments must exist in the data fields of the data objective.
        See self.data_fields for a complete list of data fields.

        Example:
        _x = np.ones((1, 3))
        _u = np.ones((1, 2))
        data.update('_x': _x, '_u': _u)

        or:
        data.update('_x': _x)
        data.update('_u': _u)

        Alternatively:
        data_dict = {
            '_x':np.ones((1, 3)),
            '_u':np.ones((1, 2))
        }

        data.update(**data_dict)


        :param **kwargs: Arbitrary number of key word arguments for data fields that should be updated.
        :type casadi.DM or numpy.ndarray

        :raises assertion: Keyword must be in existing data_fields.

        :return: None
        """
        for key, value in kwargs.items():
            assert key in self.data_fields, 'Cannot update non existing key {} in data object.'.format(key)
            if type(value) == structure3.DMStruct:
                value = value.cat
            if type(value) == DM:
                # Convert to numpy
                value = value.full()
            elif type(value) in [float, int]:
                value = np.array(value)
            # Get current results array for the given key:
            arr = getattr(self, key)
            # Append current value to results array:
            updated = np.append(arr, value.reshape(1,-1))
            # Update results array:
            setattr(self, key, updated)

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
