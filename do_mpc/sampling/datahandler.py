
import types
import pickle
import os
import numpy as np
import pathlib
import pdb
import scipy.io as sio
import copy
from do_mpc.tools import load_pickle, save_pickle



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
