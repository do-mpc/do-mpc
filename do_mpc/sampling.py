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

class Sampler:
    """The **do-mpc** Sampler class.

    """
    def __init__(self, sampling_plan):
        assert isinstance(sampling_plan, dict), 'sampling_plan must be a dict'
        assert isinstance(sampling_plan['sampling_plan'], list), 'sampling_plan must contain key list with list'
        assert np.all([isinstance(plan_i, dict) for plan_i in sampling_plan['sampling_plan']]), 'All elements of sampling plan must be a dictionary.'

        self.flags = {
        }

        # Parameters that can be set for the MHE:
        self.data_fields = [
            'save_dir',
            'overwrite_results',
        ]

        self.save_dir = './results/'
        self.overwrite_results = False


    def set_sample_function(self,sample_function):
        """
        Set sample generating function.
        """
        self.sample_function = sample_function


    def sample_data(self):
        for i, sample in enumerate(self.sampling_plan):
            result = self.sample_function(**sample)
            #pdb.set_trace()

            # if os.path.isfile(sampling_plan_name + '.pkl'):
            #     None




class SamplingPlanner:
    """A class for generating sampling plans. These sampling plans will be used by , that can be used in return for the evaluation of the considered configuration or for machine learning.


    **Configuration and sampling plan generation:**

    Configuring and generating a sampling plan involves the following steps:

    1. Set variables which should be sampled with :py:func:`set_sampling_var`, e.g. the initial state.

    2. Generate the sampling plan with :py:func:`gen_sampling_plan`.

    The generated sampling plan

    """
    def __init__(self):
        self.sampling_vars = []

    def set_sampling_var(self, name, fun_var_pdf):
        """

        """
        assert isinstance(name, str), 'name must be str, you have {}'.format(type(name))
        assert isinstance(fun_var_pdf, (types.FunctionType, types.BuiltinFunctionType)), 'fun_var_pdf must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(fun_var_pdf))
        self.sampling_vars.append({'name':name, 'fun_var_pdf':fun_var_pdf})

    def gen_sampling_plan(self, sampling_plan_name, n_samples):
        """

        """
        assert isinstance(sampling_plan_name, str), 'sampling_plan_name must be str, you have {}'.format(type(var_type))
        assert isinstance(n_samples, int), 'n_samples must be int, you have {}'.format(type(n_samples))

        sampling_plan = []

        for i in range(n_samples):

            n_digits = len(str(n_samples))

            temp_dic = {var['name']: var['fun_var_pdf']() for var in self.sampling_vars}
            temp_dic.update{'id': str(i).zfill(n_digits)}

            sampling_plan.append(temp_dic)

        # safe sampling plan
        self.sampling_plan = {'name':sampling_plan_name,'n_samples':n_samples,'sampling_plan':sampling_plan}

        # cnt = 0
        # if os.path.isfile(sampling_plan_name + '.pkl'):
        #     sampling_plan_name =
        with open(sampling_plan_name + '.pkl', 'wb') as f:
            pickle.dump(self.sampling_plan, f)

        return self.sampling_plan
