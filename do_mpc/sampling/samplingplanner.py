import types
import pickle
import os
import numpy as np
import pathlib
import pdb
import scipy.io as sio
import copy
from do_mpc.tools import load_pickle, save_pickle


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
