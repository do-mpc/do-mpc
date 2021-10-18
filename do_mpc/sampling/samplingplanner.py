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
    These sampling plans will be executed by :py:class:`do_mpc.sampling.Sampler` to generate data
    which can be used for evaluating the performance the performance of the considered configuration, machine learning, etc.

    **Configuration and sampling plan generation:**

    Configuring and generating a sampling plan involves the following steps:

    1. Set variables which should be sampled with :py:func:`set_sampling_var`.

    2. (Optional) Set further options of the SamplingPlanner with :py:meth:`set_param`

    You can then manually or automatically or as a mix of both create the sampling plan with:

    3. Generate the sampling plan with :py:func:`gen_sampling_plan`.

    """
    def __init__(self):
        self.sampling_vars = []
        self.sampling_var_names = []
        self.sampling_plan = []

        # Parameters that can be set for the SamplingPlanner:
        self.data_fields = [
            'overwrite',
            'save_format'
        ]

        self.overwrite = False
        self.save_format = 'pickle'
        self.id_precision = 3

    def set_param(self, **kwargs):
        """Set the parameters of the :py:class:`SamplingPlanner` class. Parameters must be passed as pairs of valid keywords and respective argument.
        For example:

        ::

            sp.set_param(overwrite = True)

        It is also possible and convenient to pass a dictionary with multiple parameters simultaneously as shown in the following example:

        ::

            setup_dict = {
                'overwrite': True,
                'save_format': pickle,
            }
            sp.set_param(**setup_dict)

        This makes use of thy python "unpack" operator. See `more details here`_.

        .. _`more details here`: https://codeyarns.github.io/tech/2012-04-25-unpack-operator-in-python.html

        .. note:: Default parameters are available for most settings.

        .. note:: :py:func:`set_param` can be called multiple times. Previously passed arguments are overwritten by successive calls.

        The following parameters are available:

        :param overwrite: Overwrites existing samplingplan under the same name, if set to ``True``.
        :type overwrite: bool

        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for SamplingPlanner.'.format(key))
            else:
                setattr(self, key, value)

    def set_sampling_var(self, name, fun_var_pdf=None):
        """Introduce new sampling variables to the :py:class:`SamplingPlanner`. Define variable name.
        Optionally add a function to generate values for the sampled variable (e.g. following some distribution).
        The parameter ``fun_var_pdf`` defaults to ``None`` and allows

        and the function that generates a value for the corresponding variable.

        :param name: Declare the name of the variable
        :type name: string
        :param fun_var_pdf: Declare the function of the
        :type fun_var_pdf: Function of BuiltinFunction_or_method

        :raises assertion: name must be string
        :raises assertion: must be Function or BuiltinFunction_or_Method
        """
        assert isinstance(name, str), 'name must be str, you have {}'.format(type(name))
        assert isinstance(fun_var_pdf, (types.FunctionType, types.BuiltinFunctionType, type(None))), 'fun_var_pdf must be either Function or BuiltinFunction_or_Method or None, you have {}'.format(type(fun_var_pdf))
        self.sampling_vars.append({'name':name, 'fun_var_pdf':fun_var_pdf})
        self.sampling_var_names.append(name)

    def add_sampling_case(self, **kwargs):
        """

        """
        # Create each sampling case as dict:
        temp_dic = {}
        # Iterate over all the key value pairs added in the method call:
        for key, value in kwargs.items():
            if not (key in self.sampling_var_names):
                raise Exception('{} is not a valid sampling variable. Introduce sampling variables with set_sampling_var.'.format(key))
            else:
                temp_dic[key] = value

        # Add key value pairs for all keys that were not referenced in add_sampling_case:
        for var in self.sampling_vars:
            if var['name'] not in kwargs.keys():
                temp_dic[var['name']] = var['fun_var_pdf']()

        # Generate string ID of sampling case based on index and pad with zeros:
        id = len(self.sampling_plan)
        temp_dic['id'] = str(id).zfill(self.id_precision)

        self.sampling_plan.append(temp_dic)

        return self.sampling_plan



    def gen_sampling_plan(self, n_samples):
        """Generate the sampling plan. The generated sampling contains ``n_samples`` samples based on the defined variables and the corresponding evaluation functions.

        :param n_samples: The number of generated samples
        :type n_samples: int

        :raises assertion: n_samples must be int

        :return: Returns the newly created sampling plan.
        :rtype: list
        """
        assert isinstance(n_samples, int), 'n_samples must be int, you have {}'.format(type(n_samples))

        for i in range(n_samples):
            self.add_sampling_case()


        return self.sampling_plan


    def export(self, sampling_plan_name):
        """

        """
        if not os.path.isfile(sampling_plan_name + '.pkl') or self.overwrite:
            save_pickle(sampling_plan_name, self.sampling_plan)
        else:
            for i in range(1,10000):
                if not os.path.isfile(sampling_plan_name + '_' + str(i) + '.pkl'):
                    self.sampling_plan.update({'name': sampling_plan_name + '_' + str(i)})
                    save_pickle(sampling_plan_name + '_' + str(i), self.sampling_plan)
                    break
