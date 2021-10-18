
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
            'sample_name',
            'save_format'
        ]

        self.data_dir = None
        self.sample_name = 'sample'
        self.save_format = 'pickle'

        self.sampling_plan = sampling_plan
        self.sampling_vars = list(sampling_plan[0].keys())
        self.post_processing = {}

        self.pre_loaded_data = {}


    def __getitem__(self, ind):
        """ Filter data from the DataHandler. Pass the desired indices via slicing and returns the post processed data.
        """
        assert self.flags['set_post_processing'], 'No post processing function is set. Cannot query data.'
        assert self.data_dir is not None, 'Must set data_dir before querying items.'

        val = self._init_results()

        # For each sample:
        for i, sample in enumerate(self.sampling_plan[ind]):

            # Check if this result was previously loaded. If not, add it to the dict of pre_loaded_data.
            # pdb.set_trace()
            result = self._lazy_loading(sample)

            val = self._post_process_single(val,sample,result)

        return val


    def _init_results(self):
        """Private method: initializes the resulting dictionary such that all sampling_vars and post processed data is included
        """
        val = {key: [] for key in self.sampling_vars}
        val.update({key: [] for key in self.post_processing.keys()})

        return val


    def _lazy_loading(self,sample):
        """ Private method: Chekcs if data is already loaded to reduce the computational load.
        """
        if sample['id'] in self.pre_loaded_data.keys():
            result = self.pre_loaded_data[sample['id']]
        else:
            result = self._load(sample['id'])
            self.pre_loaded_data.update({sample['id']: result})

        return result


    def _post_process_single(self,val,sample,result):
        """ Private method: Applies all post processing functions to a single sample and stores them.
        """
        # Post process result
        for key in self.post_processing:
            val[key].append(self.post_processing[key](result))

        for var_i in self.sampling_vars:
            val[var_i].append(sample[var_i])

        return val


    def filter(self, filter_fun):
        """ Filter data from the DataHandler. Pass the desired filters

        :param filter_fun: The name of the sampling plan.
        :type filter: Function or BuiltinFunction_or_Method

        :raises assertion: No post processing function is set
        :raises assertion: filter_fun must be either Function of BuiltinFunction_or_Method

        :return: Returns the post processed samples that satisfy the filter
        :rtype: dict
        """
        assert self.flags['set_post_processing'], 'No post processing function is set. Cannot query data.'
        assert isinstance(filter_fun, (types.FunctionType, types.BuiltinFunctionType)), 'post_processing_function must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(filter_fun))

        val = self._init_results()

        # Wrapper to ensure arbitrary arguments are accepted
        def wrap_fun(**kwargs):
            return filter_fun(**{arg_i: kwargs[arg_i] for arg_i in filter_fun.__code__.co_varnames})

        # For each sample:
        for i, sample in enumerate(self.sampling_plan):

            if wrap_fun(**sample)==True:
                # Check if this result was previously loaded. If not, add it to the dict of pre_loaded_data.
                result = self._lazy_loading(sample)

                val = self._post_process_single(val,sample,result)

        return val


    def _load(self, sample_id):
        """ Private method: Load data generated from a sampling plan, either '.pkl' or '.mat'
        """
        name = '{sample_name}_{id}'.format(sample_name=self.sample_name, id=sample_id)

        if self.save_format == 'pickle':
            load_name = self.data_dir + name + '.pkl'
        elif self.save_format == 'mat':
            load_name = self.data_dir + name+'.mat'

        if self.save_format == 'pickle':
            result = load_pickle(load_name)
        elif self.save_format == 'mat':
            result = sio.loadmat(load_name)

        return result


    def set_param(self, **kwargs):
        """Set the parameters of the DataHandler
        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for DataHandler.'.format(key))
            else:
                setattr(self, key, value)


    def set_post_processing(self, name, post_processing_function):
        """Set a post processing function. The generated sampling contains ``n_samples`` samples based on the defined variables and the corresponding evaluation functions.

        :param name: The name of the sampling plan.
        :type name: string
        :param post_processing_function: The post processing function to be evaluted
        :type n_samples: Function or BuiltinFunction_or_Method

        :raises assertion: name must be string
        :raises assertion: post_processing_function must be either Function of BuiltinFunction_or_Method
        """
        assert isinstance(name, str), 'name must be str, you have {}'.format(type(name))
        assert isinstance(post_processing_function, (types.FunctionType, types.BuiltinFunctionType)), 'post_processing_function must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(post_processing_function))

        self.post_processing.update({name: post_processing_function})
        self.flags['set_post_processing'] = True
