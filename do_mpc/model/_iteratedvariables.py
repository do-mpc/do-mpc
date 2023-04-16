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
import casadi.tools as castools
import pdb


class IteratedVariables:
    """ Class to initiate properties and attributes for iterated variables.
    This class is inherited to all iterating **do-mpc** classes and based on the :py:class:`Model`.

    Warnings:
        This base class can not be used independently.
    """

    def __init__(self):
        assert 'model' in self.__dict__.keys(), 'Cannot initialize variables before assigning the model to the current class instance.'

        # Initialize structure for intial conditions:
        self._x0 = self.model._x(0.0)
        self._u0 = self.model._u(0.0)
        self._z0 = self.model._z(0.0)
        self._t0 = np.array([0.0])


    def _convert2struct(self, val, struct):
        """ Convert array to structure.
        Pass ``val`` which can be an int, float, array, structure and return
        a numerical structure based on the second argument ``structure``.

        If a structure is passed, return the structure unchanged.

        Performs some sanity checks.
        """

        # convert to array
        if isinstance(val, (float, int)):
            val = np.array([val])

        # Check dimensions
        err_msg = 'Variable cannot be set because the supplied vector has the wrong size. You have {} and the model is setup for {}'
        n_val = np.prod(val.shape)
        n_var = struct.size
        assert n_val == n_var, err_msg.format(n_val, n_var)

        # Convert to structure (or return structure)
        if isinstance(val, (np.ndarray, castools.DM)):
            val = struct(val)
        elif isinstance(val, castools.structure3.DMStruct):
            pass
        else:
            types = (np.ndarray, castools.DM, castools.structure3.DMStruct)
            raise Exception('x0 must be of tpye {}. You have: {}'.format(types, type(val)))

        return val

    @property
    def x0(self):
        """ Initial state and current iterate.
        This is the numerical structure holding the information about the current states
        in the class. The property can be indexed according to the model definition.

        **Example:**

        ::

            model = do_mpc.model.Model('continuous')
            model.set_variable('_x','temperature', shape=(4,1))

            ...
            mhe = do_mpc.estimator.MHE(model)
            # or
            mpc = do_mpc.estimator.MPC(model)

            # Get or set current value of variable:
            mpc.x0['temperature', 0] # 0th element of variable
            mpc.x0['temperature']    # all elements of variable
            mpc.x0['temperature', 0:2]    # 0th and 1st element

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``

        """
        return self._x0

    @x0.setter
    def x0(self, val):
        self._x0 = self._convert2struct(val, self.model._x)

    @property
    def u0(self):
        """ Initial input and current iterate.
        This is the numerical structure holding the information about the current input
        in the class. The property can be indexed according to the model definition.

        **Example:**

        ::

            model = do_mpc.model.Model('continuous')
            model.set_variable('_u','heating', shape=(4,1))

            ...
            mhe = do_mpc.estimator.MHE(model)
            # or
            mpc = do_mpc.estimator.MPC(model)

            # Get or set current value of variable:
            mpc.u0['heating', 0] # 0th element of variable
            mpc.u0['heating']    # all elements of variable
            mpc.u0['heating', 0:2]    # 0th and 1st element

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``

        """
        return self._u0

    @u0.setter
    def u0(self, val):
        self._u0 = self._convert2struct(val, self.model._u)

    @property
    def z0(self):
        """ Initial algebraic state and current iterate.
        This is the numerical structure holding the information about the current algebraic states
        in the class. The property can be indexed according to the model definition.

        **Example:**

        ::

            model = do_mpc.model.Model('continuous')
            model.set_variable('_z','temperature', shape=(4,1))

            ...
            mhe = do_mpc.estimator.MHE(model)
            # or
            mpc = do_mpc.estimator.MPC(model)

            # Get or set current value of variable:
            mpc.z0['temperature', 0] # 0th element of variable
            mpc.z0['temperature']    # all elements of variable
            mpc.z0['temperature', 0:2]    # 0th and 1st element

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``

        """
        return self._z0

    @z0.setter
    def z0(self, val):
        self._z0 = self._convert2struct(val, self.model._z)

    @property
    def t0(self):
        """ Current time marker of the class.
        Use this property to set of query the time.

        Set with ``int``, ``float``, ``numpy.ndarray`` or ``casadi.DM`` type.
        """
        return self._t0

    @t0.setter
    def t0(self,val):
        if isinstance(val, (int,float)):
            self._t0 = np.array([val])
        elif isinstance(val, np.ndarray):
            assert val.size == 1, 'Cant set time with shape {}. Must contain exactly one element.'.format(val.size)
            self._t0 = val.flatten()
        elif isinstance(val, castools.DM):
            assert val.size == 1, 'Cant set time with shape {}. Must contain exactly one element.'.format(val.size)
            self._t0 = val.full().flatten()
        else:
            types = (np.ndarray, float, int, castools.DM)
            raise Exception('Passing object of type {} to set the current time. Must be of type {}'.format(type(val), types))