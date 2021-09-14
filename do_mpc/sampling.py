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


class Sampler:
    """The **do-mpc** Sampler class.

    """
    def __init__(self, sampling_plan):
        assert isinstance(sampling_plan, dict), 'sampling_plan must be a dict'
        assert isinstance(sampling_plan['sampling_plan'], list), 'sampling_plan must contain key list with list'
        assert np.all([isinstance(plan_i, dict) for plan_i in sampling_plan['sampling_plan']]), 'All elements of sampling plan must be a dictionary.'

        self.sampling_plan = sampling_plan

        self.flags = {
        }

        # Parameters that can be set for the Sampler:
        self.data_fields = [
            'save_dir',
            'overwrite_results',
            'save_format'
        ]

        self.save_dir = './{}/'.format(sampling_plan['name'])
        self.overwrite_results = False
        self.save_format = 'pickle'

    @property
    def save_dir(self):
        """Set the save directory for the results.
        If the directory does not exist yet, it is created. This is also possible for nested structures.
        """
        return self._save_dir

    @save_dir.setter
    def save_dir(self, val):
        self._save_dir = val
        pathlib.Path(val).mkdir(parents=True, exist_ok=True)

    def set_param(self, **kwargs):
        """

        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for Sampler.'.format(key))
            else:
                setattr(self, key, value)


    def set_sample_function(self,sample_function):
        """
        Set sample generating function.
        """
        self.sample_function = sample_function

    def _save(self, sample_id, result):
        name = '{plan_name}_{id}'.format(plan_name=self.sampling_plan['name'], id=sample_id)

        if self.save_format == 'pickle':
            save_name = self.save_dir + name + '.pkl'
        elif self.save_format == 'mat':
            save_name = self.save_dir + name+'.mat'

        if os.path.isfile(save_name):
            None
        else:
            if self.save_format == 'pickle':
                with open(save_name, 'wb') as f:
                    pickle.dump(result, f)
            elif self.save_format == 'mat':
                sio.savemat(save_name, {name: result})



    def sample_data(self):
        for i, sample in enumerate(self.sampling_plan['sampling_plan']):

            # Pop sample id from dictionary (not an argument to the sample function)
            sample_id = sample.pop('id')

            # Call sample function to create sample (pass sample information)
            result = self.sample_function(**sample)

            self._save(sample_id, result)





class SamplingPlanner:
    """A class for generating sampling plans. These sampling plans will be executed by :py:class:`Sampler` to generate data which can be used for evaluating the performance the performance of the considered configuration, machine learning, etc.


    **Configuration and sampling plan generation:**

    Configuring and generating a sampling plan involves the following steps:

    1. Set variables which should be sampled with :py:func:`set_sampling_var`, e.g. the initial state.

    2. Generate the sampling plan with :py:func:`gen_sampling_plan`.

    """
    def __init__(self):
        self.sampling_vars = []

        # Parameters that can be set for the SamplingPlanner:
        self.data_fields = [
            'overwrite'
        ]

        self.overwrite = False

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

        # save sampling plan (if necessary with unique numbering)
        self.sampling_plan = {'n_samples':n_samples,'sampling_plan':sampling_plan}

        if not os.path.isfile(sampling_plan_name + '.pkl') or self.overwrite:
            with open(sampling_plan_name + '.pkl', 'wb') as f:
                pickle.dump(self.sampling_plan, f)
            self.sampling_plan.update({'name': sampling_plan_name})
        else:
            for i in range(1,10000):
                if not os.path.isfile(sampling_plan_name + '_' + str(i) + '.pkl'):
                    with open(sampling_plan_name + '_' + str(i) + '.pkl', 'wb') as f:
                        pickle.dump(self.sampling_plan, f)
                    self.sampling_plan.update({'name': sampling_plan_name + '_' + str(i)})
                    break


        return self.sampling_plan