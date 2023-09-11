
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
import logging
from inspect import signature
from typing import Union


class DataHandler:
    """Post-processing data created from a sampling plan.
    Data (individual samples) were created with :py:class:`do_mpc.sampling.Sampler`.
    The list of all samples originates from :py:class:`do_mpc.sampling.SamplingPlanner` and is used to
    initiate this class (``sampling_plan``).

    The class can be created with optional keyword arguments which are passed to :py:meth:`set_param`.

    **Configuration and retrieving processed data:**

    1. Initiate the object with the ``sampling_plan`` originating from :py:class:`do_mpc.sampling.SamplingPlanner`.

    2. Set parameters with :py:meth:`set_param`. Most importantly, the directory in which the individual samples are located should be passe with ``data_dir`` argument.

    3. (Optional) set one (or multiple) post-processing functions. These functions are applied to each loaded sample and can, e.g., extract or compile important information.

    4. Load and return samples either by indexing with the :py:meth:`__getitem__` method or by filtering with :py:meth:`filter`.

    **Example:**

    ::

        sp = do_mpc.sampling.SamplingPlanner()

        # Plan with two variables alpha and beta:
        sp.set_sampling_var('alpha', np.random.randn)
        sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

        plan = sp.gen_sampling_plan(n_samples=10)

        sampler = do_mpc.sampling.Sampler(plan)

        # Sampler computes the product of two variables alpha and beta
        # that were created in the SamplingPlanner:

        def sample_function(alpha, beta):
            return alpha*beta

        sampler.set_sample_function(sample_function)

        sampler.sample_data()

        # Create DataHandler object with same plan:
        dh = do_mpc.sampling.DataHandler(plan)

        # Assume you want to compute the square of the result of each sample
        dh.set_post_processing('square', lambda res: res**2)

        # As well as the value itself:
        dh.set_post_processing('default', lambda res: res)

        # Query all post-processed results with:
        dh[:]

    """
    def __init__(self, sampling_plan, **kwargs):
        self.flags = {
            'set_post_processing' : False,
        }

        # Parameters that can be set for the DataHandler:
        self.data_fields = [
            'data_dir',
            'sample_name',
            'save_format'
        ]

        self.data_dir = './'
        self.sample_name = 'sample'
        self.save_format = 'pickle'

        self.sampling_plan = sampling_plan
        self.sampling_vars = list(sampling_plan[0].keys())
        self.post_processing = {}

        self.pre_loaded_data = {'id':[], 'data':[]}

        if kwargs:
            self.set_param(**kwargs)


    @property
    def data_dir(self):
        """Set the directory where the results are stored.
        """
        return self._data_dir

    @data_dir.setter
    def data_dir(self, val):
        self._data_dir = val

    def __getitem__(self, ind):
        """ Index results from the :py:class:`DataHandler`. Pass an index or a slice operator.
        """

        try:
            if isinstance(ind, int):
                samples = [self.sampling_plan[ind]]
            elif isinstance(ind, (tuple, slice, list)):
                samples = self.sampling_plan[ind]
            else:
                raise Exception('ind must be of type int, tuple, slice or list. You have {}'.format(type(ind)))
        except IndexError:
            print('---------------------------------------------------------------')
            print('Trying to access a non-existent element from the sampling plan.')
            print('---------------------------------------------------------------')
            raise

        return_list = []

        # For each sample:
        for sample in samples:
            # Check if this result was previously loaded. If not, add it to the dict of pre_loaded_data.
            result = self._lazy_loading(sample)
            result_processed = self._post_process_single(sample,result)
            return_list.append(result_processed)

        return return_list

    def _lazy_loading(self,sample):
        """ Private method: Checks if data is already loaded to reduce the computational load.
        """
        if sample['id'] in self.pre_loaded_data['id']:
            ind = self.pre_loaded_data['id'].index(sample['id'])
            result = self.pre_loaded_data['data'][ind]
        else:
            result = self._load(sample['id'])
            self.pre_loaded_data['id'].append(sample['id'])
            self.pre_loaded_data['data'].append(result)

        return result


    def _post_process_single(self,sample,result):
        """ Private method: Applies all post processing functions to a single sample and stores them.
        """
        result_processed = copy.copy(sample)
        # Post process result
        if self.flags['set_post_processing']:
            for key in self.post_processing:
                if result is not None:
                    # post_processing function is either just a function of the result or of the sample and the result.
                    if self.post_processing[key]['n_args'] == 1:
                        result_processed[key] = self.post_processing[key]['function'](result)
                    elif self.post_processing[key]['n_args'] == 2:
                        result_processed[key] = self.post_processing[key]['function'](sample, result)
                else:
                    result_processed[key] = None
        # Result without post processing
        else:
            result_processed['res'] = result

        return result_processed


    def filter(self, 
               input_filter:Union[types.FunctionType,types.BuiltinFunctionType]=None, 
               output_filter:Union[types.FunctionType,types.BuiltinFunctionType]=None
               )->list:
        """ Filter data from the DataHandler. Filters can be applied to inputs or to results that were obtained with the post-processing functions.
        Filtering returns only a subset from the created samples based on arbitrary conditions.

        **Example**:

        ::

            sp = do_mpc.sampling.SamplingPlanner()

            # SamplingPlanner with two variables alpha and beta:
            sp.set_sampling_var('alpha', np.random.randn)
            sp.set_sampling_var('beta', lambda: np.random.randint(0,5))
            plan = sp.gen_sampling_plan()

            ...

            dh = do_mpc.sampling.DataHandler(plan)
            dh.set_post_processing('square', lambda res: res**2)

            # Return all samples with alpha < 0 and beta > 2
            dh.filter(input_filter = lambda alpha, beta: alpha < 0 and beta > 2)
            # Return all samples for which the computed value square < 5
            dh.filter(output_filter = lambda square: square < 5)

        Args:
            input_filter: Function to filter the data.
            output_filter: Function to filter the data

        Raises:
            assertion: No post processing function is set
            assertion: filter_fun must be either Function of BuiltinFunction_or_Method

        Returns:
            Returns the post processed samples that satisfy the filter
        """
        assert isinstance(input_filter, (types.FunctionType, types.BuiltinFunctionType, type(None))), 'input_filter must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(input_filter))
        assert isinstance(output_filter, (types.FunctionType, types.BuiltinFunctionType, type(None))), 'output_filter must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(output_filter))

        return_list = []

        # Wrapper to ensure arbitrary arguments are accepted
        def wrap_fun_in(**kwargs):
            if input_filter is None:
                return True
            else:
                return input_filter(**{arg_i: kwargs[arg_i] for arg_i in input_filter.__code__.co_varnames})

        # Wrapper to ensure arbitrary arguments are accepted
        def wrap_fun_out(**kwargs):
            if output_filter is None:
                return True
            else:
                return output_filter(**{arg_i: kwargs[arg_i] for arg_i in output_filter.__code__.co_varnames})

        # For each sample:
        for sample in self.sampling_plan:
            if wrap_fun_in(**sample)==True:
                # Check if this result was previously loaded. If not, add it to the dict of pre_loaded_data.
                result = self._lazy_loading(sample)
                result_processed = self._post_process_single(sample,result)
                # Check if the computed post-processing value satsifies the output_filter condition:
                if wrap_fun_out(**result_processed)==True:
                    return_list.append(result_processed)

        return return_list


    def _load(self, sample_id):
        """ Private method: Load data generated from a sampling plan, either '.pkl' or '.mat'
        """
        name = '{sample_name}_{id}'.format(sample_name=self.sample_name, id=sample_id)

        if self.save_format == 'pickle':
            load_name = self.data_dir + name + '.pkl'
        elif self.save_format == 'mat':
            load_name = self.data_dir + name+'.mat'

        try:
            if self.save_format == 'pickle':
                result = load_pickle(load_name)
            elif self.save_format == 'mat':
                result = sio.loadmat(load_name)
        except FileNotFoundError:
            logging.warning('Could not find or load file: {}. Check data_dir parameter and make sure sample has already been generated.'.format(load_name))
            result = None

        return result


    def set_param(self, **kwargs)->None:
        """Set the parameters of the DataHandler.

        Parameters must be passed as pairs of valid keywords and respective argument.
        For example:

        ::

            datahandler.set_param(overwrite = True)

        Args:
            data_dir(bool): Directory where the data can be found (as defined in the :py:class:`do_mpc.sampling.Sampler`).
            sample_name(str): Naming scheme for samples (as defined in the :py:class:`do_mpc.sampling.Sampler`).
            save_format(str): Choose either ``pickle`` or ``mat`` (as defined in the :py:class:`do_mpc.sampling.Sampler`).
        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for DataHandler.'.format(key))
            else:
                setattr(self, key, value)


    def set_post_processing(self, name:str, post_processing_function:Union[types.FunctionType,types.BuiltinFunctionType])->None:
        """Set a post processing function.
        The post processing function is applied to all loaded samples, e.g. with :py:meth:`__getitem__` or :py:meth:`filter`.
        Users can set an arbitrary amount of post processing functions by repeatedly calling this method.

        The ``post_processing_function`` can have two possible signatures:

        1. ``post_processing_function(case_definition, sample_result)``

        2. ``post_processing_function(sample_result)``

        Where ``case_definition`` is a ``dict`` of all variables introduced in the :py:class:`do_mpc.sampling.SamplingPlanner`
        and ``sample_results`` is the result obtained from the function introduced with :py:class:`do_mpc.sampling.Sampler.set_sample_function`.

        Note:
            Setting a post processing function with an already existing name will overwrite the previously set post processing function.

        **Example:**

        ::

            sp = do_mpc.sampling.SamplingPlanner()

            # Plan with two variables alpha and beta:
            sp.set_sampling_var('alpha', np.random.randn)
            sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

            plan = sp.gen_sampling_plan(n_samples=10)

            sampler = do_mpc.sampling.Sampler(plan)

            # Sampler computes the product of two variables alpha and beta
            # that were created in the SamplingPlanner:

            def sample_function(alpha, beta):
                return alpha*beta

            sampler.set_sample_function(sample_function)

            sampler.sample_data()

            # Create DataHandler object with same plan:
            dh = do_mpc.sampling.DataHandler(plan)

            # Assume you want to compute the square of the result of each sample
            dh.set_post_processing('square', lambda res: res**2)

            # As well as the value itself:
            dh.set_post_processing('default', lambda res: res)

            # Query all post-processed results with:
            dh[:]

        Args:
            name: Name of the output of the post-processing operation
            post_processing_function: The post processing function to be evaluted

        Raises:
            assertion: name must be string
            assertion: post_processing_function must be either Function of BuiltinFunction
        """
        assert isinstance(name, str), 'name must be str, you have {}'.format(type(name))
        assert isinstance(post_processing_function, (types.FunctionType, types.BuiltinFunctionType)), 'post_processing_function must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(post_processing_function))

        # Check signature of function for number of arguments.
        sig = signature(post_processing_function)
        n_args = len(sig.parameters.keys())

        self.post_processing.update({name: {'function': post_processing_function, 'n_args': n_args}})
        self.flags['set_post_processing'] = True