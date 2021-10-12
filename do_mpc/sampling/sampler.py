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


class Sampler:
    """The **do-mpc** Sampler class.

    """
    def __init__(self, sampling_plan):
        assert isinstance(sampling_plan, dict), 'sampling_plan must be a dict'
        assert isinstance(sampling_plan['sampling_plan'], list), 'sampling_plan must contain key list with list'
        assert np.all([isinstance(plan_i, dict) for plan_i in sampling_plan['sampling_plan']]), 'All elements of sampling plan must be a dictionary.'

        self.sampling_plan = sampling_plan
        self.sampling_vars = sampling_plan['sampling_plan'][0].keys()

        self.flags = {
            'set_sample_function': False,
        }

        # Parameters that can be set for the Sampler:
        self.data_fields = [
            'data_dir',
            'overwrite',
        ]

        self.data_dir = './{}/'.format(sampling_plan['name'])
        self.overwrite = False

    @property
    def data_dir(self):
        """Set the save directory for the results.
        If the directory does not exist yet, it is created. This is also possible for nested structures.
        """
        return self._data_dir

    @data_dir.setter
    def data_dir(self, val):
        self._data_dir = val
        pathlib.Path(val).mkdir(parents=True, exist_ok=True)

    def set_param(self, **kwargs):
        """Configure the :py:class:Sampler`` class.

        Parameters must be passed as pairs of valid keywords and respective argument.
        For example:

        ::

            sampler.set_param(overwrite = True)


        :param overwrite: Should previously created results be overwritten. Default is ``False`
        :type overwrite: bool

        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for Sampler.'.format(key))
            else:
                setattr(self, key, value)


    def set_sample_function(self, sample_function):
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

        :param sample_function: Function to create each sample of the sampling plan.
        :type sample_function: FunctionType

        """
        assert isinstance(sample_function, (types.FunctionType, types.BuiltinFunctionType)), 'sample_function must be a function'
        dset = set(inspect.getargspec(sample_function).args) - set(self.sampling_vars)
        assert len(dset) == 0, 'sample_function must only contain keyword arguments that appear as sample vars in the sampling_plan. You have the unknown arguments: {}'.format(dset)

        self.sample_function = sample_function

        self.flags['set_sample_function'] = True


    def _save_name(self, sample_id):
        """Private method. Used in :py:meth:`sample_data`.
        Creates the name for a given sample based on the sample plan name and the sample id.
        """
        name = '{plan_name}_{id}'.format(plan_name=self.sampling_plan['name'], id=sample_id)

        if self.sampling_plan['save_format'] == 'pickle':
            save_name = self.data_dir + name + '.pkl'
        elif self.sampling_plan['save_format'] == 'mat':
            save_name = self.data_dir + name+'.mat'

        return save_name


    def _save(self, save_name, result):
        """Private method. Saves the result for a single sample in the defined format.
        Considers the ``overwrite`` parameter to check if existing results should be overwritten.
        """
        if os.path.isfile(save_name) or self.overwrite:
            None
        else:
            if self.sampling_plan['save_format'] == 'pickle':
                save_pickle(save_name, result)
            elif self.sampling_plan['save_format'] == 'mat':
                sio.savemat(save_name, {name: result})


    def sample_data(self):
        """Sample data after having configured the :py:class:`Sampler`.
        No user input is required and the method will iterate through all the items defined in the ``sampling_plan`` (obtained with :py:class:SamplingPlanner).
        Depending on your ``sample_function`` (set with :py:meth:`set_sample_function`) and the total number of samples, executing this method may take some time.
        """
        assert self.flags['set_sample_function'], 'Cannot sample before setting the sample function with Sampler.set_sample_function'

        for i, sample in enumerate(self.sampling_plan['sampling_plan']):

            # Pop sample id from dictionary (not an argument to the sample function)
            sample_i = copy.copy(sample)
            sample_id = sample_i.pop('id')


            # Create and safe result if sample result does not exist:
            save_name = self._save_name(sample_id)
            if not os.path.isfile(save_name) or self.overwrite:

                # Call sample function to create sample (pass sample information)
                result = self.sample_function(**sample_i)

                self._save(save_name, result)

            printProgressBar(i+1, self.sampling_plan['n_samples'], prefix = 'Progress:', suffix = 'Complete', length = 50)
