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

import types
import pickle
import os
import numpy as np
import pathlib
import pdb
import scipy.io as sio
import copy

def save_pickle(filename, data):
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path_to_file):
    with open(path_to_file, 'rb') as f:
        data = pickle.load(f)
    return data

class Sampler:
    """The **do-mpc** Sampler class.

    """
    def __init__(self, sampling_plan):
        assert isinstance(sampling_plan, dict), 'sampling_plan must be a dict'
        assert isinstance(sampling_plan['sampling_plan'], list), 'sampling_plan must contain key list with list'
        assert np.all([isinstance(plan_i, dict) for plan_i in sampling_plan['sampling_plan']]), 'All elements of sampling plan must be a dictionary.'

        self.sampling_plan = sampling_plan
        self.sampling_vars = sampling_plan['sampling_plan'][0].keys()

        self.flags = {
            'set_sample_function': False,
        }

        # Parameters that can be set for the Sampler:
        self.data_fields = [
            'data_dir',
            'overwrite',
        ]

        self.data_dir = './{}/'.format(sampling_plan['name'])
        self.overwrite = False

    @property
    def data_dir(self):
        """Set the save directory for the results.
        If the directory does not exist yet, it is created. This is also possible for nested structures.
        """
        return self._data_dir

    @data_dir.setter
    def data_dir(self, val):
        self._data_dir = val
        pathlib.Path(val).mkdir(parents=True, exist_ok=True)

    def set_param(self, **kwargs):
        """

        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for Sampler.'.format(key))
            else:
                setattr(self, key, value)


    def set_sample_function(self, sample_function):
        """
        Set sample generating function.
        The sampling function produces a sample result for each sample definition in the ``sampling_plan``.

        It is important that the sample function only uses keyword arguments with the same name as previously defined in the ``sampling_plan``.

        **Example:**

        ::

            sp = do_mpc.sampling.SamplingPlanner()

            sp.set_sampling_var('alpha', np.random.randn)
            sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

            sampler = do_mpc.sampling.Sampler(plan)

            def sample_function(alpha, beta, gamma):
                return alpha*beta

            sampler.set_sample_function(sample_function)

        :param sample_function: Function to create each sample of the sampling plan.
        :type sample_function: FunctionType

        """
        assert isinstance(sample_function, (types.FunctionType, types.BuiltinFunctionType)), 'sample_function must be a function'
        dset = set(sample_function.__code__.co_varnames) - set(self.sampling_vars)
        assert len(dset) == 0, 'sample_function must only contain keyword arguments that appear as sample vars in the sampling_plan. You have the unknown arguments: {}'.format(dset)

        self.sample_function = sample_function

        self.flags['set_sample_function'] = True


    def _save_name(self, sample_id):
        name = '{plan_name}_{id}'.format(plan_name=self.sampling_plan['name'], id=sample_id)

        if self.sampling_plan['save_format'] == 'pickle':
            save_name = self.data_dir + name + '.pkl'
        elif self.sampling_plan['save_format'] == 'mat':
            save_name = self.data_dir + name+'.mat'

        return save_name


    def _save(self, save_name, result):
        if os.path.isfile(save_name):
            None
        else:
            if self.sampling_plan['save_format'] == 'pickle':
                save_pickle(save_name, result)
            elif self.sampling_plan['save_format'] == 'mat':
                sio.savemat(save_name, {name: result})


    def sample_data(self):
        assert self.flags['set_sample_function'], 'Cannot sample before setting the sample function with Sampler.set_sample_function'

        for i, sample in enumerate(self.sampling_plan['sampling_plan']):

            # Pop sample id from dictionary (not an argument to the sample function)
            sample_i = copy.copy(sample)
            sample_id = sample_i.pop('id')


            # Create and safe result if sample result does not exist:
            save_name = self._save_name(sample_id)
            if not os.path.isfile(save_name) or self.overwrite:

                # Call sample function to create sample (pass sample information)
                result = self.sample_function(**sample_i)

                self._save(save_name, result)





class SamplingPlanner:
    """A class for generating sampling plans.
    These sampling plans will be executed by :py:class:`Sampler` to generate data
    which can be used for evaluating the performance the performance of the considered configuration, machine learning, etc.

    **Configuration and sampling plan generation:**

    Configuring and generating a sampling plan involves the following steps:

    1. Set variables which should be sampled with :py:func:`set_sampling_var`, e.g. the initial state.

    2. (Optional) Set further options of the SamplingPlanner with :py:meth:`set_param`

    3. Generate the sampling plan with :py:func:`gen_sampling_plan`.

    """
    def __init__(self):
        self.sampling_vars = []

        # Parameters that can be set for the SamplingPlanner:
        self.data_fields = [
            'overwrite',
            'save_format'
        ]

        self.overwrite = False
        self.save_format = 'pickle'

    def set_param(self, **kwargs):
        """

        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for SamplingPlanner.'.format(key))
            else:
                setattr(self, key, value)

    def set_sampling_var(self, name, fun_var_pdf):
        """Introduce new sampling variables to the :py:class:`SamplingPlanner`. Define variable name and the function that generates a value for the corresponding variable.

        :param name: Declare the name of the variable
        :type name: string
        :param fun_var_pdf: Declare the function of the
        :type fun_var_pdf: Function of BuiltinFunction_or_method

        :raises assertion: name must be string
        :raises assertion: must be Function or BuiltinFunction_or_Method
        """
        assert isinstance(name, str), 'name must be str, you have {}'.format(type(name))
        assert isinstance(fun_var_pdf, (types.FunctionType, types.BuiltinFunctionType)), 'fun_var_pdf must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(fun_var_pdf))
        self.sampling_vars.append({'name':name, 'fun_var_pdf':fun_var_pdf})

    def gen_sampling_plan(self, sampling_plan_name, n_samples):
        """Generate the sampling plan. The generated sampling contains ``n_samples`` samples based on the defined variables and the corresponding evaluation functions.

        :param sampling_plan_name: The name of the sampling plan.
        :type sampling_plan_name: string
        :param n_samples: The number generated samples
        :type n_samples: int

        :raises assertion: sampling_plan_name must be string
        :raises assertion: n_samples must be int

        :return: Returns the newly created sampling plan.
        :rtype: list
        """
        assert isinstance(sampling_plan_name, str), 'sampling_plan_name must be str, you have {}'.format(type(var_type))
        assert isinstance(n_samples, int), 'n_samples must be int, you have {}'.format(type(n_samples))

        sampling_plan = []

        for i in range(n_samples):

            n_digits = len(str(n_samples))

            temp_dic = {var['name']: var['fun_var_pdf']() for var in self.sampling_vars}
            temp_dic.update({'id': str(i).zfill(n_digits)})

            sampling_plan.append(temp_dic)

        # save sampling plan
        self.sampling_plan = {'n_samples':n_samples, 'save_format': self.save_format, 'sampling_plan':sampling_plan}

        if not os.path.isfile(sampling_plan_name + '.pkl') or self.overwrite:
            self.sampling_plan.update({'name': sampling_plan_name})
            save_pickle(sampling_plan_name, self.sampling_plan)
        else:
            for i in range(1,10000):
                if not os.path.isfile(sampling_plan_name + '_' + str(i) + '.pkl'):
                    self.sampling_plan.update({'name': sampling_plan_name + '_' + str(i)})
                    save_pickle(sampling_plan_name + '_' + str(i), self.sampling_plan)
                    break


        return self.sampling_plan


class DataHandler:

    def __init__(self, sampling_plan):

        self.flags = {
            'set_compilation_function' : False,
        }

        # Parameters that can be set for the DataHandler:
        self.data_fields = [
            'data_dir',
        ]

        self.data_dir = './{}/'.format(sampling_plan['name'])

        self.sampling_plan = sampling_plan
        self.sampling_vars = sampling_plan['sampling_plan'][0].keys()


    def __getitem__(self, ind_fun):
        """

        """
        assert self.flags['set_compilation_function'], 'No compilation function is set. Cannot query data.'

        val = {key:[] for key in self.sampling_vars}
        val.update({'result':[]})


        # Ensure tuple
        if not isinstance(ind_fun, tuple):
            ind_fun = (ind_fun,)

        # Initiate array for loading
        load_bool = np.zeros(self.sampling_plan['n_samples'])

        # For each sample:
        for i, sample in enumerate(self.sampling_plan['sampling_plan']):
            load_sample = True

            # Check all conditions
            for ind_fun_i in ind_fun:
                # Wrapper to ensure arbitrary arguments are accepted
                def wrap_fun(**kwargs):
                    return ind_fun_i(**{arg_i: kwargs[arg_i] for arg_i in ind_fun_i.__code__.co_varnames})

                if wrap_fun(**sample) == False:
                    load_sample = False
                    break

            if load_sample:
                if 'result' in sample.keys():
                    compiled_result = self.compilation_function(sample['result'])
                else:
                    result = self._load(sample['id'])
                    compiled_result = self.compilation_function(result)

                val['result'].append(compiled_result)
                for var_i in self.sampling_vars:
                    val[var_i].append(sample[var_i])

        return val

    def _load(self, sample_id):
        name = '{plan_name}_{id}'.format(plan_name=self.sampling_plan['name'], id=sample_id)

        if self.sampling_plan['save_format'] == 'pickle':
            load_name = self.data_dir + name + '.pkl'
        elif self.sampling_plan['save_format'] == 'mat':
            load_name = self.data_dir + name+'.mat'


        if self.sampling_plan['save_format'] == 'pickle':
            with open(load_name, 'rb') as f:
                result = pickle.load(f)
        elif self.sampling_plan['save_format'] == 'mat':
            result = sio.loadmat(load_name)

        return result

    def set_param(self, **kwargs):
        """

        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for DataHandler.'.format(key))
            else:
                setattr(self, key, value)

    def set_compilation_function(self, compilation_function):
        """

        """
        self.compilation_function = compilation_function
        self.flags['set_compilation_function'] = True
