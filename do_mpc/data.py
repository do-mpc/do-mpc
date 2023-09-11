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

"""
Storage and handling of data.
"""

import numpy as np
import casadi.tools as castools
import pdb
import pickle
import do_mpc
import os
from typing import Union,Tuple,Dict

class Data:
    """**do-mpc** data container. An instance of this class is created for the active **do-mpc** classes,
    e.g. :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.estimator.MHE`.

    The class is initialized with an instance of the :py:class:`do_mpc.model.Model` which contains all
    information about variables (e.g. states, inputs etc.).

    The :py:class:`Data` class has a public API but is mostly used by other **do-mpc** classes, e.g. updated in the ``.make_step`` calls.

    Args:
        model: model object from the :py:class:`do_mpc.model` 
    """
    def __init__(self, model:Union[do_mpc.model.Model,do_mpc.model.LinearModel]):
        self.dtype = 'default'
        assert model.flags['setup'] == True, 'Model was not setup. After the complete model creation call model.setup().'
        # As discussed here: https://groups.google.com/forum/#!topic/casadi-users/dqAb4tnA2ik
        # struct_SX cannot be unpickled (seems like a bug)
        # TODO: Find better workaround.
        self.model = model.__dict__.copy()
        self.model.pop('_rhs')
        self.model.pop('_aux_expression')
        self.model.pop('_y_expression')
        self.model.pop('_alg')
        self.model.pop('sv')


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

        # Accelerate __getitem__ calls (to retrieve results) by saving indices of previous queries.
        self.result_queries = {'ind':[], 'f_ind':[]}

    def __getitem__(self, ind:Tuple)->np.ndarray:
        """Query data fields. This method can be used to obtain the stored results in the :py:class:`Data` instance.

        The full list of available fields can be inspected with:

        ::

            print(data.data_fields)

        The dict also denotes the dimension of each field.

        The method allows for power indexing the results for the fields
        ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``, ``_aux``, ``_y``
        where further indices refer to the configured variables in the :py:class:`do_mpc.model.Model` and :py:class:`do_mpc.model.LinearModel` instance.

        **Example:**

        ::

            # Assume the following model was used (excerpt):
            model = do_mpc.model.Model('continuous')

            model.set_variable('_x', 'Temperature', shape=(5,1)) # Vector
            model.set_variable('_p', 'disturbance', shape=(3,3)) # Matrix
            model.set_variable('_u', 'heating')                  # scalar

            ...

            # the model was used (among others) for the MPC controller
            mpc = do_mpc.controller.MPC(model)

            ...

            # Query the mpc.data instance:
            mpc.data['_x']                      # Return all states
            mpc.data['_x', 'Temperature']       # Return the 5 temp states
            mpc.data['_x', 'Temperature', :2]   # Return the first 2 temp. states
            mpc.data['_p', 'disturbance', 0, 2] # Matrix allows for further indices

            # Other fields can also be queried, e.g.:
            mpc.data['_time']                   # current time
            mpc.data['t_wall_total']            # optimizer runtime
            # These do not allow further indices.

        Args:
            ind: Power index to query the prediction of a specific variable.
        
        Returns:
            Returns the queried data field (for all time instances)
        """
        # ensure list:
        if not isinstance(ind, tuple):
            ind = [ind]

        # First element is the data_field:
        data_field = ind[0]
        # Check validity:
        keys = self.data_fields.keys()
        assert data_field in keys, 'Your queried variable {} is not available. Please choose from {}'.format(data_field, keys)

        if len(ind)>1:
            # If further indices exist:
            if ind in self.result_queries['ind']:
                i = self.result_queries['ind'].index(ind)
                f_ind = self.result_queries['f_ind'][i]
            else:
                powerind = ind[1:]
                f_ind = self.model[data_field].f[powerind]
                self.result_queries['ind'].append(ind)
                self.result_queries['f_ind'].append(f_ind)
            out = getattr(self, data_field)[:, f_ind]

        else:
            # If not just return the field:
            out = getattr(self, data_field)
        return out

    def init_storage(self)->None:
        """Create new (empty) arrays for all variables.
        The variables of interest are listed in the ``data_fields`` dictionary,
        with their respective dimension. This dictionary may be updated.
        The :py:class:`do_mpc.controller.MPC` class adds for example optimizer information.
        """
        for field_i, dim_i in self.data_fields.items():
            setattr(self, field_i, np.empty((0, dim_i)))

    def set_meta(self, **kwargs)->None:
        """Set meta data for the current instance of the data object.
        """
        for key, value in kwargs.items():
            self.meta_data[key] = value

    def update(self, **kwargs:Dict[np.ndarray,castools.DM])->None:
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

        Args:
            kwargs : Arbitrary number of key word arguments for data fields that should be updated.

        Raises:
            assertion: Keyword must be in existing data_fields.
        """
        for key, value in kwargs.items():
            assert key in self.data_fields.keys(), 'Cannot update non existing key {} in data object.'.format(key)
            if type(value) == castools.structure3.DMStruct:
                value = value.cat
            if type(value) == castools.DM:
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

    def export(self)->dict:
        """The export method returns a dictionary of the stored data.

        Returns:
            Dictionary of the currently stored data.
        """
        export_dict = {field_name: getattr(self, field_name) for field_name in self.data_fields}
        return export_dict


class MPCData(Data):
    """**do-mpc** data container for the :py:class:`do_mpc.controller.MPC` instance.
    This method inherits from :py:class:`Data` and extends it to query the MPC predictions.
    
    Warning:
        For robust multi-stage MPC, the :py:class:`MPCData` class stores by default only the nominal values of the uncertain parameters. 

    Args:
        model: model from :py:class:`do_mpc.model`
    """

    def __init__(self, model:Union[do_mpc.model.Model,do_mpc.model.LinearModel]):
        super().__init__(model)
        # Accelerate prediction calls by saving indices of previous queries.
        self.prediction_queries = {'ind':[], 'f_ind':[]}

    def prediction(self, ind:tuple, t_ind:float=-1)->np.ndarray:
        """Query the MPC trajectories.
        Use this method to obtain specific MPC trajectories from the data object.

        Warnings:
            This method requires that the optimal solution is stored in the :py:class:`do_mpc.data.MPCData` instance.
            Storing the optimal solution must be activated with :py:func:`do_mpc.controller.MPC.set_param`.


        Querying predicted trajectories requires the use of power indices, which is passed as tuple e.g.:

        ::

            data.prediction((var_type, var_name, i), t_ind)

        where

        * ``var_type`` refers to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``, ``_aux``

        * ``var_name`` refers to the user-defined names in the :py:class:`do_mpc.model.Model`

        * Use ``i`` to index vector valued variables.

        The method returns a multidimensional numpy.ndarray. The dimensions refer to:

        ::

            arr = data.prediction(('_x', 'x_1'))
            arr.shape
            >> (n_size, n_horizon, n_scenario)

        with:

        * ``n_size`` denoting the number of elements in ``x_1``, where ``n_size = 1`` is a scalar variable.

        * ``n_horizon`` is the MPC horizon defined with :py:func:`do_mpc.controller.MPC.set_param`

        * ``n_scenario`` refers to the number of uncertain scenarios (for robust MPC).

        Additional to the power index tuple, a time index (``t_ind``) can be passed to access the prediction for a certain
        time.

        Args:
            ind: Power index to query the prediction of a specific variable.
            t_ind: Time index

        Returns:
            Predicted trajectories for the queries variable.
        """

        assert self.meta_data['store_full_solution'], 'Optimal trajectory is not stored. Please update your MPC settings.'
        assert isinstance(ind, tuple), 'Query index must be of type tuple.'

        structure_scenario = self.meta_data['structure_scenario']

        if self._opt_x_num.shape[0]==0:
            _opt_x_num = np.zeros((1,self.opt_x.shape[0]))
            _opt_p_num = np.zeros((1,self.opt_p.shape[0]))
            _opt_aux_num = np.zeros((1,self.opt_aux.shape[0]))
        else:
            _opt_x_num = self._opt_x_num
            _opt_p_num = self.opt_p_num
            _opt_aux_num = self._opt_aux_num

        if ind[0] in ['_x', '_z']:
            if ind in self.prediction_queries['ind']:
                i = self.prediction_queries['ind'].index(ind)
                f_ind = self.prediction_queries['f_ind'][i]
            else:
                f_ind = self.opt_x.f[(ind[0], slice(None), lambda v: castools.horzcat(*v),slice(None), -1)+ind[1:]]
                f_ind = np.array([f_ind_k.full() for f_ind_k in f_ind], dtype='int32')
                # sort pred such that each column belongs to one scenario
                # - By indexing structure_scenario until f_ind.shape[0] we cover the case of _x and _z at the same time
                f_ind = f_ind[range(f_ind.shape[0]),:,structure_scenario[:f_ind.shape[0],:].T].T
                # Store f_ind:
                self.prediction_queries['ind'].append(ind)
                self.prediction_queries['f_ind'].append(f_ind)
            out = _opt_x_num[t_ind,f_ind]

        elif ind[0] =='_u':
            if ind in self.prediction_queries['ind']:
                i = self.prediction_queries['ind'].index(ind)
                f_ind = self.prediction_queries['f_ind'][i]
            else:
                f_ind = self.opt_x.f[(ind[0], slice(None), lambda v: castools.horzcat(*v),slice(None))+ind[1:]]
                f_ind = np.array([f_ind_k.full() for f_ind_k in f_ind], dtype='int32')
                # sort pred such that each column belongs to one scenario
                if self.meta_data['open_loop']:
                    f_ind = f_ind[range(f_ind.shape[0]),:,structure_scenario[:-1,[0]].T].T
                else:
                    f_ind = f_ind[range(f_ind.shape[0]),:,structure_scenario[:-1,:].T].T

                # Store f_ind:
                self.prediction_queries['ind'].append(ind)
                self.prediction_queries['f_ind'].append(f_ind)
            out = _opt_x_num[t_ind,f_ind]

        elif ind[0]=='_tvp':
            if ind in self.prediction_queries['ind']:
                i = self.prediction_queries['ind'].index(ind)
                f_ind = self.prediction_queries['f_ind'][i]
            else:
                f_ind = self.opt_p.f[(ind[0], slice(None))+ind[1:]]
                f_ind = np.array(f_ind).reshape(1, -1, 1)
                #f_ind = np.array([f_ind_k.full() for f_ind_k in f_ind], dtype='int32')
                # Store f_ind:
                self.prediction_queries['ind'].append(ind)
                self.prediction_queries['f_ind'].append(f_ind)
            out = _opt_p_num[t_ind,f_ind]

        elif ind[0]=='_aux':
            if ind in self.prediction_queries['ind']:
                i = self.prediction_queries['ind'].index(ind)
                f_ind = self.prediction_queries['f_ind'][i]
            else:
                f_ind = self.opt_aux.f[(ind[0], slice(None), lambda v: castools.horzcat(*v),slice(None))+ind[1:]]
                f_ind = np.array([f_ind_k.full() for f_ind_k in f_ind], dtype='int32')
                # sort pred such that each column belongs to one scenario
                f_ind = f_ind[range(f_ind.shape[0]),:,structure_scenario[:-1,:].T].T
                # Store f_ind:
                self.prediction_queries['ind'].append(ind)
                self.prediction_queries['f_ind'].append(f_ind)
            out = _opt_aux_num[t_ind,f_ind]
        else:
            raise ValueError('Index {} not recognized.'.format(ind[0]))

        return out



def save_results(save_list:list, 
                 result_name:str='results', 
                 result_path:str='./results/', 
                 overwrite:bool=False
                 )->None:
    """Exports the data objects from the **do-mpc** modules in ``save_list`` as a pickled file. Supply any, all or a selection of (as a list):

    * :py:class:`do_mpc.controller.MPC`

    * :py:class:`do_mpc.simulator.Simulator`

    * :py:class:`do_mpc.estimator.Estimator`

    These objects can be used in post-processing to create graphics with the :py:class:`do_mpc.graphics_backend`.

    Args:
        save_list: List of the objects to be stored.
        result_name: Name of the result file, defaults to 'result'.
        result_path: Result path, defaults to './results/'.
        overwrite: Option to overwrite existing results, defaults to False. Index will be appended if file already exists.

    Raises:
        assertion: save_list must be a list.
        assertion: result_name must be a string.
        assertion: results_path must be a string.
        assertion: overwrite must be boolean.
        Exception: save_list contains object which is neither do_mpc simulator, optimizizer nor estimator.
    """

    assert isinstance(save_list, list), 'save_list must be a list.'
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

def load_results(file_name:str)->Dict:
    """ Simple wrapper to open and unpickle a file.
    If used for **do-mpc** results, this will return a dictionary with the stored **do-mpc** modules:

    * :py:class:`do_mpc.controller.MPC`

    * :py:class:`do_mpc.simulator.Simulator`

    * :py:class:`do_mpc.estimator.Estimator`

    Args:
        file_name : File name (including path) for the file to be opened and unpickled.

    Returns:
        Returns the results stored in .pkl file.
    """

    with open(file_name, 'rb') as f:
        results = pickle.load(f)

    return results
