
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



class DataHandler:
    def __init__(self, sampling_plan):
        """Post-processing data created from a sampling plan.
        Data (individual samples) were created with :py:class:`do_mpc.sampling.sampler.Sampler`.
        The list of all samples originates from :py:class:`do_mpc.sampling.samplingplanner.SamplingPlanner` and is used to
        initiate this class (``sampling_plan``).

        **Configuration and retrieving processed data:**

        1. Initiate the object with the ``sampling_plan`` originating from :py:class:`do_mpc.sampling.samplingplanner.SamplingPlanner`.

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
        """ Index results from the :py:class:`DataHandler`. Pass an index or a slice operator.
        """
        assert self.data_dir is not None, 'Must set data_dir before querying items.'

        val = self._init_results()

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

        # For each sample:
        for sample in samples:
            # Check if this result was previously loaded. If not, add it to the dict of pre_loaded_data.
            result = self._lazy_loading(sample)

            val = self._post_process_single(val,sample,result)

        return val


    def _init_results(self):
        """Private method: initializes the resulting dictionary such that all sampling_vars and post processed data is included.
        """
        val = {key: [] for key in self.sampling_vars}
        if self.flags['set_post_processing']:
            val.update({key: [] for key in self.post_processing.keys()})
        else:
            val.update({'res': []})

        return val


    def _lazy_loading(self,sample):
        """ Private method: Checks if data is already loaded to reduce the computational load.
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
        if self.flags['set_post_processing']:
            for key in self.post_processing:
                if result is not None:
                    val[key].append(self.post_processing[key](result))
                else:
                    val[key].append(result)
        # Result without post processing
        else:
            val['res'] = result

        for var_i in self.sampling_vars:
            val[var_i].append(sample[var_i])

        return val


    def filter(self, filter_fun):
        """ Filter data from the DataHandler. Pass the desired filters.
        This allows to return only a subset from the created samples based on arbitrary conditions applied to the variables in the sampling plan.

        **Example**:

        ::

            sp = do_mpc.sampling.SamplingPlanner()

            # SamplingPlanner with two variables alpha and beta:
            sp.set_sampling_var('alpha', np.random.randn)
            sp.set_sampling_var('beta', lambda: np.random.randint(0,5))
            plan = sp.gen_sampling_plan()

            ...

            dh = do_mpc.sampling.DataHandler(plan)
            # Return all samples with alpha < 0 and beta > 2
            dh.filter(lambda alpha, beta: alpha < 0 and beta > 2)

        .. note::

            Filters are only applied to inputs (defined in the ``sampling_plan``) of the generated data.


        :param filter_fun: Function  to filter the data.
        :type filter: Function or BuiltinFunction_or_Method

        :raises assertion: No post processing function is set
        :raises assertion: filter_fun must be either Function of BuiltinFunction_or_Method

        :return: Returns the post processed samples that satisfy the filter
        :rtype: dict
        """
        assert isinstance(filter_fun, (types.FunctionType, types.BuiltinFunctionType)), 'post_processing_function must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(filter_fun))

        val = self._init_results()

        # Wrapper to ensure arbitrary arguments are accepted
        def wrap_fun(**kwargs):
            return filter_fun(**{arg_i: kwargs[arg_i] for arg_i in filter_fun.__code__.co_varnames})

        # For each sample:
        for sample in self.sampling_plan:
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

        try:
            if self.save_format == 'pickle':
                result = load_pickle(load_name)
            elif self.save_format == 'mat':
                result = sio.loadmat(load_name)
        except FileNotFoundError:
            logging.warning('Could not find or load file: {}. Check data_dir parameter and make sure sample has already been generated.'.format(load_name))
            result = None

        return result


    def set_param(self, **kwargs):
        """Set the parameters of the DataHandler.

        Parameters must be passed as pairs of valid keywords and respective argument.
        For example:

        ::

            datahandler.set_param(overwrite = True)


        :param data_dir: Directory where the data can be found (as defined in the :py:clas:`do_mpc.sampling.sampler.Sampler`).
        :type data_dir: bool

        :param sample_name: Naming scheme for samples (as defined in the :py:clas:`do_mpc.sampling.sampler.Sampler`).
        :type sample_name: str

        :save_format: Choose either ``pickle`` or ``mat`` (as defined in the :py:clas:`do_mpc.sampling.sampler.Sampler`).
        :type save_format: str

        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for DataHandler.'.format(key))
            else:
                setattr(self, key, value)


    def set_post_processing(self, name, post_processing_function):
        """Set a post processing function.
        The post processing function is applied to all loaded samples, e.g. with :py:meth:`__getitem__` or :py:meth:`filter`.
        Users can set an arbitrary amount of post processing functions by repeatedly calling this method.

        ..note::
        
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

        :param name: Name of the output of the post-processing operation
        :type name: string
        :param post_processing_function: The post processing function to be evaluted
        :type n_samples: Function or BuiltinFunction

        :raises assertion: name must be string
        :raises assertion: post_processing_function must be either Function of BuiltinFunction
        """
        assert isinstance(name, str), 'name must be str, you have {}'.format(type(name))
        assert isinstance(post_processing_function, (types.FunctionType, types.BuiltinFunctionType)), 'post_processing_function must be either Function or BuiltinFunction_or_Method, you have {}'.format(type(post_processing_function))

        self.post_processing.update({name: post_processing_function})
        self.flags['set_post_processing'] = True
