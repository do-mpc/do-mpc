
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
        """ Filter data from the DataHandler. Pass

        """
        assert self.flags['set_post_processing'], 'No post processing function is set. Cannot query data.'

        val = self._init_results()

        # For each sample:
        for i, sample in enumerate(self.sampling_plan['sampling_plan'][ind]):

            # Check if this result was previously loaded. If not, add it to the dict of pre_loaded_data.
            # pdb.set_trace()
            result = self._lazy_loading(sample)

            val = self._post_process_single(val,sample,result)

        return val


    def _init_results(self):

        val = {key: [] for key in self.sampling_vars}
        val.update({key: [] for key in self.post_processing.keys()})

        return val


    def _lazy_loading(self,sample):

        if sample['id'] in self.pre_loaded_data.keys():
            result = self.pre_loaded_data[sample['id']]
        else:
            result = self._load(sample['id'])
            self.pre_loaded_data.update({sample['id']: result})

        return result


    def _post_process_single(self,val,sample,result):

        # Post process result
        for key in self.post_processing:
            val[key].append(self.post_processing[key](result))

        for var_i in self.sampling_vars:
            val[var_i].append(sample[var_i])

        return val


    def filter(self, filter_fun):
        """ Filter data from the DataHandler. Pass

        """
        assert self.flags['set_post_processing'], 'No post processing function is set. Cannot query data.'

        val = self._init_results()

        # Wrapper to ensure arbitrary arguments are accepted
        def wrap_fun(**kwargs):
            return filter_fun(**{arg_i: kwargs[arg_i] for arg_i in filter_fun.__code__.co_varnames})

        # For each sample:
        for i, sample in enumerate(self.sampling_plan['sampling_plan']):

            if wrap_fun(**sample)==True:
                # Check if this result was previously loaded. If not, add it to the dict of pre_loaded_data.
                result = self._lazy_loading(sample)

                val = self._post_process_single(val,sample,result)

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
