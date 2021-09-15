
import types
import pickle
import os
import numpy as np
import pathlib
import pdb
import scipy.io as sio
import copy
from do_mpc.tools import load_pickle, save_pickle
import types



class DataHandler:

    def __init__(self, sampling_plan):

        self.flags = {
            'set_post_processing' : False,
        }

        # Parameters that can be set for the DataHandler:
        self.data_fields = [
            'data_dir',
        ]

        self.data_dir = './{}/'.format(sampling_plan['name'])

        self.sampling_plan = sampling_plan
        self.sampling_vars = sampling_plan['sampling_plan'][0].keys()
        self.post_processing = {}

        self.pre_loaded_data = {}


    def __getitem__(self, ind):





    def filter(self, filter_fun):
        """ Filter data from the DataHandler. Pass

        """
        assert self.flags['set_post_processing'], 'No compilation function is set. Cannot query data.'

        val = {key:[] for key in self.sampling_vars}
        val.update({'result':[]})

        # Wrapper to ensure arbitrary arguments are accepted
        def wrap_fun(**kwargs):
            return filter_fun(**{arg_i: kwargs[arg_i] for arg_i in filter_fun.__code__.co_varnames})

        # For each sample:
        for i, sample in enumerate(self.sampling_plan['sampling_plan']):

            if wrap_fun(**sample)==True:
                # Check if this result was previously loaded. If not, add it to the dict of pre_loaded_data.
                if sample['id'] in self.pre_loaded_data.keys():
                    result = self.pre_loaded_data['id']
                else:
                    result = self._load(sample['id'])
                    self.pre_loaded_data.update({sample['id']: result})

                # Compile result
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
            result = load_pickle(load_name)
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

    def set_post_processing(self, name, post_processing_function):
        """

        """
        assert isinstance(name, str), 'name must be str, you have {}'.format(type(name))
        assert isinstance(post_processing_function, (types.FunctionType, types.BuiltinFunctionType)), 'post_processing_function must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(post_processing_function))

        self.post_processing.update({name: post_processing_function})
        self.flags['set_post_processing'] = True
