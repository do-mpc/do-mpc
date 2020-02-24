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
import pickle
import do_mpc


class Data:
    """**do-mpc** data container. An instance of this class is created for all active **do-mpc** classes,
    e.g. :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC`, :py:class:`do_mpc.estimator.MHE`.

    The class is initialized with an instance of the :py:class:`do_mpc.model.Model` which contains all
    information about variables (e.g. states, inputs etc.).

    The :py:class:`Data` class has a public API but is mostly used by other **do-mpc** classes, e.g. updated in the ``.make_step`` calls.
    """
    def __init__(self, model):
        self.dtype = 'default'
        assert model.flags['setup'] == True, 'Model was not setup. After the complete model creation call model.setup_model().'
        # As discussed here: https://groups.google.com/forum/#!topic/casadi-users/dqAb4tnA2ik
        # struct_SX cannot be unpickled (seems like a bug)
        # TODO: Find better workaround.
        self.model = model.__dict__.copy()
        self.model.pop('_rhs')
        self.model.pop('_aux_expression')
        self.model.pop('_y_expression')


        # TODO: n_aux not existing
        #self._aux = np.empty((0, model.n_aux))
        # Dictionary with possible data_fields in the class and their respective dimension. All data is numpy ndarray.
        self.data_fields = {
            '_time': 1,
            '_x':    model.n_x,
            '_y':    model.n_y,
            '_u':    model.n_u,
            '_z':    model.n_z,
            '_tvp':  model.n_tvp,
            '_p':    model.n_p,
            '_aux':  model.n_aux,
        }

        self.init_storage()
        self.meta_data = {}

    def init_storage(self):
        """Create new (empty) arrays for all variables.
        The variables of interest are listed in the ``data_fields`` dictionary,
        with their respective dimension. This dictionary may be updated.
        The :py:class:`do_mpc.controller.MPC` class adds for example optimizer information.
        """
        for field_i, dim_i in self.data_fields.items():
            setattr(self, field_i, np.empty((0, dim_i)))

    def set_meta(self, **kwargs):
        """Set meta data for the current instance of the data object.
        """
        for key, value in kwargs.items():
            self.meta_data[key] = value

    def update(self, **kwargs):
        """Update value(s) of the data structure with key word arguments.
        These key word arguments must exist in the data fields of the data objective.
        See self.data_fields for a complete list of data fields.

        Example:

        ::

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


        :param kwargs: Arbitrary number of key word arguments for data fields that should be updated.
        :type kwargs: casadi.DM or numpy.ndarray

        :raises assertion: Keyword must be in existing data_fields.

        :return: None
        """
        for key, value in kwargs.items():
            assert key in self.data_fields.keys(), 'Cannot update non existing key {} in data object.'.format(key)
            if type(value) == structure3.DMStruct:
                value = value.cat
            if type(value) == DM:
                # Convert to numpy
                value = value.full()
            elif type(value) in [float, int, bool]:
                value = np.array(value)
            # Get current results array for the given key:
            arr = getattr(self, key)
            # Append current value to results array:
            updated = np.append(arr, value.reshape(1,-1), axis=0)
            # Update results array:
            setattr(self, key, updated)

    def export(self):
        """The export method returns a dictionary of the stored data.

        :return: Dictionary of the currently stored data.
        :rtype: dict
        """
        export_dict = {field_name: getattr(self, field_name) for field_name in self.data_fields}
        return export_dict


def save_results(save_list, result_name='results', result_path='./results/', overwrite=False):
    """Exports the data objects from the **do-mpc** modules in ``save_list`` as a pickled file. Supply any, all or a selection of (as a list):

    * :py:class:`do_mpc.controller.MPC`

    * :py:class:`do_mpc.simulator.Simulator`

    * :py:class:`do_mpc.estimator.Estimator`

    These objects can be used in post-processing to create graphics with the :py:class:`do_mpc.graphics_backend`.

    :param save_list: List of the objects to be stored.
    :type save_list: list
    :param result_name: Name of the result file, defaults to 'result'.
    :type result_name: string, optional
    :param result_path: Result path, defaults to './results/'.
    :type result_path: string, optional
    :param overwrite: Option to overwrite existing results, defaults to False. Index will be appended if file already exists.
    :type overwrite: bool, optional

    :raises assertion: save_list must be a list.
    :raises assertion: result_name must be a string.
    :raises assertion: results_path must be a string.
    :raises assertion: overwrite must be boolean.
    :raises Exception: save_list contains object which is neither do_mpc simulator, optimizizer nor estimator.

    :return: None
    :rtype: None
    """

    assert isinstance(save_list, list), 'save_list must be a string.'
    assert isinstance(result_name, str), 'result_name must be a string.'
    assert isinstance(result_path, str), 'results_path must be a string.'
    assert isinstance(overwrite, bool), 'overwrite must be boolean.'

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results = {}

    for obj_i in save_list:
        if isinstance(obj_i, do_mpc.controller.MPC):
            results.update({'mpc': obj_i.data})
        elif isinstance(obj_i, do_mpc.simulator.Simulator):
            results.update({'simulator': obj_i.data})
        elif isinstance(obj_i, (do_mpc.estimator.StateFeedback, do_mpc.estimator.EKF, do_mpc.estimator.MHE)):
            results.update({'estimator': obj_i.data})
        else:
            raise Exception('save_list contains object which is neither do_mpc simulator, optimizizer nor estimator.')

    # Dynamically generate new result name if name is already taken in result_path.
    if overwrite==False:
        ind = 1
        ext_result_name = result_name
        while os.path.isfile(result_path+ext_result_name+'.pkl'):
            ext_result_name = '{ind:03d}_{name}'.format(ind=ind, name=result_name)
            ind += 1
        result_name = ext_result_name

    with open(result_path+result_name+'.pkl', 'wb') as f:
        pickle.dump(results, f)

def load_results(file_name):
    """ Simple wrapper to open and unpickle a file.
    If used for **do-mpc** results, this will return a dictionary with the stored **do-mpc** modules:

    * :py:class:`do_mpc.controller.MPC`

    * :py:class:`do_mpc.simulator.Simulator`

    * :py:class:`do_mpc.estimator.Estimator`

    :param file_name: File name (including path) for the file to be opened and unpickled.
    :type file_name: str
    """

    with open(file_name, 'rb') as f:
        results = pickle.load(f)

    return results
