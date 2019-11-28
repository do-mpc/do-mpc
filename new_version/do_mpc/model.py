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


class model:
    def __init__(self):
        None

    def set_variables(self, _x=None, _u=None, _z=None, _tvp=None, _p=None):
        # TODO: Add docstring.
        if _x:
            self._x = _x
            self._rhs = struct_SX(_x)
            self.n_x = _x.shape[0]
        if _u:
            self._u = _u
            self.n_u = _u.shape[0]
        if _z:
            self._z = _z
            self.n_z = _z.shape[0]
        if _tvp:
            self._tvp = _tvp
            self.n_tvp = _tvp.shape[0]
        if _p:
            self._p = _p
            self.n_p = _p.shape[0]

    def set_aux(self, **kwargs):
        # TODO: Make sure kwargs are passed correctly.
        self._aux = struct_SX([
            entry(name, expr=expr) for name, expr in kwargs.items()
        ])

    def get_variables(self):
        # TODO: Add docstring.
        return self._x, self._u, self._z, self._tvp, self._p, self._aux

    def get_rhs(self):
        # TODO: Add docstring.
        return self._rhs
