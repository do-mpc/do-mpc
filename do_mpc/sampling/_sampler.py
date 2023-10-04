import types
import pickle
import os
import inspect
import numpy as np
import pathlib
import pdb
import scipy.io as sio
import copy
from do_mpc.tools import load_pickle, save_pickle, printProgressBar
from typing import Union,Callable

class Sampler:
    """Generate samples based on a sampling plan.
    Initiate the class by passing a :py:class:`do_mpc.sampling.SamplingPlanner` (``sampling_plan``) object.
    The class can be configured to create samples based on the defined cases in the ``sampling_plan``.

    The class can be created with optional keyword arguments which are passed to :py:meth:`set_param`.

    **Configuration and sampling:**

    1. (Optional) use :py:meth:`set_param` to configure the class. Use :py:attr:`data_dir` to choose the save location for the samples.

    2. Set the sample generating function with :py:meth:`set_sample_function`. This function is executed for each of the samples in the ``sampling_plan``.

    3. Use :py:meth:`sample_data` to generate all samples defined in the ``sampling_plan``. A new file is written for each sample.

    4. **Or:** Create an individual sample result with :py:meth:`sample_idx`, where an index (``int``) referring to the ``sampling_plan`` determines the sampled case.

    Note:
        By default, the :py:class:`Sampler` will only create samples that do not already exist in the chosen :py:attr:`data_dir`.

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
    """
    def __init__(self, sampling_plan:list, **kwargs):
        assert isinstance(sampling_plan, list), 'sampling_plan must be a list'
        assert np.all([isinstance(plan_i, dict) for plan_i in sampling_plan]), 'All elements of sampling plan must be a dictionary.'

        self.sampling_plan = sampling_plan
        self.sampling_vars = list(sampling_plan[0].keys())
        self.n_samples = len(sampling_plan)
        self.completion_list = []

        self.flags = {
            'set_sample_function': False,
        }

        # Parameters that can be set for the Sampler:
        self.data_fields = [
            'overwrite',
            'sample_name',
            'save_format',
            'print_progress'
        ]

        self.data_dir = './'
        self.sample_name = 'sample'
        self.save_format = 'pickle'
        self.overwrite = False
        self.print_progress = True
        self.n_processes = 1

        if kwargs:
            self.set_param(**kwargs)

    @property
    def data_dir(self):
        """Set the save directory for the results.
        If the directory does not exist yet, it is created. If the directory is nested all (non-existing)
        parent folders are also created.

        **Example:**

        ::

            sampler = do_mpc.sampling.Sampler()
            sampler.data_dir = './samples/experiment_1/'

        This will set the directory to the indicated path. If the path does not exist, all folders are created.
        """
        return self._data_dir

    @data_dir.setter
    def data_dir(self, val):
        self._data_dir = val
        pathlib.Path(val).mkdir(parents=True, exist_ok=True)

    def set_param(self, **kwargs)->None:
        """Configure the :py:class:`do_mpc.sampling.Sampler` class.

        Parameters must be passed as pairs of valid keywords and respective argument.
        For example:

        ::

            sampler.set_param(overwrite = True)

        Args:
            overwrite(bool): Should previously created results be overwritten. Default is ``False``
            sample_name(str): Naming scheme for samples.
            save_format(str): Choose either ``pickle`` or ``mat``.
            print_progress(bool): Print progress-bar to terminal. Default is ``True``.
        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for Sampler.'.format(key))
            else:
                setattr(self, key, value)


    def set_sample_function(self, 
                            sample_function:Callable[[Union[types.FunctionType,types.BuiltinFunctionType],
                                                      Union[types.FunctionType,types.BuiltinFunctionType]],
                                                     Union[types.FunctionType,types.BuiltinFunctionType]]
                            )->None:
        """
        Set sample generating function.
        The sampling function produces a sample result for each sample definition in the ``sampling_plan``
        and is called in the method :py:meth:`sample_data`.

        It is important that the sample function only uses keyword arguments **with the same name as previously defined** in the ``sampling_plan``.

        **Example:**

        ::

            sp = do_mpc.sampling.SamplingPlanner()

            sp.set_sampling_var('alpha', np.random.randn)
            sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

            sampler = do_mpc.sampling.Sampler(plan)

            def sample_function(alpha, beta):
                return alpha*beta

            sampler.set_sample_function(sample_function)

        Args:
            sample_function: Function to create each sample of the sampling plan.
        """
        assert isinstance(sample_function, (types.FunctionType, types.BuiltinFunctionType)), 'sample_function must be a function'
        dset = set(inspect.getfullargspec(sample_function).args) - set(self.sampling_vars)
        assert len(dset) == 0, 'sample_function must only contain keyword arguments that appear as sample vars in the sampling_plan. You have the unknown arguments: {}'.format(dset)

        self.sample_function = sample_function

        self.flags['set_sample_function'] = True


    def _save_name(self, sample_id):
        """Private method. Used in :py:meth:`sample_data`.
        Creates the name for a given sample based on the sample plan name and the sample id.
        """
        name = '{sample_name}_{id}'.format(sample_name=self.sample_name, id=sample_id)

        if self.save_format == 'pickle':
            save_name = name + '.pkl'
        elif self.save_format == 'mat':
            save_name = name+'.mat'

        return save_name


    def _save(self, save_name, result):
        """Private method. Saves the result for a single sample in the defined format.
        Considers the ``overwrite`` parameter to check if existing results should be overwritten.
        """
        if not os.path.isfile(self.data_dir + save_name) or self.overwrite:
            if self.save_format == 'pickle':
                save_pickle(self.data_dir + save_name, result)
            elif self.save_format == 'mat':
                sio.savemat(self.data_dir + save_name, {'res': result})


    def sample_idx(self, idx:int)->None:
        """Sample case based on the index of the sample.

        Args:
            idx: Index of the ``sampling_plan`` for which the sample should be created.

        Raises:
            assertion: Index must be between 0 and ``n_samples``.
            assertion: sample_function must be set prior to sampling data.
        """
        #assert isinstance(idx, int), 'idx must be of type index'
        assert self.flags['set_sample_function'], 'Cannot sample before setting the sample function with Sampler.set_sample_function'
        assert idx>=0 and idx<=len(self.sampling_plan), 'Invalid value for idx. Must be between 0 and {}. You have {}'.format(len(self.sampling_plan), idx)

        # Pop sample id from dictionary (not an argument to the sample function)
        sample_i = copy.copy(self.sampling_plan[idx])
        sample_id = sample_i.pop('id')

        # Create and safe result if sample result does not exist:
        save_name = self._save_name(sample_id)
        if not os.path.isfile(self.data_dir + save_name) or self.overwrite:
            # Call sample function to create sample (pass sample information)
            result = self.sample_function(**sample_i)
            # Save  true results:
            self._save(save_name, result)
            # Add id to completion list:
            self.completion_list.append(sample_id)

        if self.print_progress:
            printProgressBar(len(self.completion_list), self.n_samples, prefix = 'Progress:', suffix = 'Complete', length = 50)


    def sample_data(self)->None:
        """Sample data after having configured the :py:class:`Sampler`.
        No user input is required and the method will iterate through all the items defined in the ``sampling_plan``
        (obtained with :py:class:`do_mpc.sampling.SamplingPlanner`).

        Note:
            Depending on your ``sample_function`` (set with :py:meth:`set_sample_function`) and the total number of samples, executing this method may take some time.

        Note:
            If ``sampler.set_param(overwrite = False)`` (default) data will only be sampled for instances that do not yet exist.
        """
        for i, _ in enumerate(self.sampling_plan):
            self.sample_idx(i)