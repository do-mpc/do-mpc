import types
import pickle
import os
import numpy as np
import pathlib
import pdb
import scipy.io as sio
import copy
import itertools
from do_mpc.tools import load_pickle, save_pickle
from typing import Union,Callable

class SamplingPlanner:
    """A class for generating sampling plans.
    These sampling plans will be executed by :py:class:`do_mpc.sampling.Sampler` to generate data.

    The class can be created with optional keyword arguments which are passed to :py:meth:`set_param`.

    **Configuration and sampling plan generation:**

    1. Set variables which should be sampled with :py:func:`set_sampling_var`.

    2. (Optional) Set further options of the SamplingPlanner with :py:meth:`set_param`

    3. Generate the sampling plan with :py:func:`gen_sampling_plan`.

    4. And / or: Add specific sampling case with :py:meth:`add_sampling_case`.

    5. Export the plan with all sampling cases with :py:meth:`export`
    """
    def __init__(self, **kwargs):
        self.sampling_vars = []
        self.sampling_var_names = []
        self.sampling_plan = []

        # Parameters that can be set for the SamplingPlanner:
        self.data_fields = [
            'overwrite',
            'id_precision',
        ]

        self.data_dir = './'
        self.overwrite = False
        self.id_precision = 3

        if kwargs:
            self.set_param(**kwargs)

    @property
    def data_dir(self):
        """Set the save directory for the ``samplingplan``.
        If the directory does not exist yet, it is created. If the directory is nested all (non-existing)
        parent folders are also created.

        **Example:**

        ::

            sp = do_mpc.sampling.SamplingPlanner()
            sp.data_dir = './samples/experiment_1/'

        This will set the directory to the indicated path. If the path does not exist, all folders are created.
        """
        return self._data_dir

    @data_dir.setter
    def data_dir(self, val):
        self._data_dir = val
        pathlib.Path(val).mkdir(parents=True, exist_ok=True)

    def set_param(self, **kwargs)->None:
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

        Note:
            :py:func:`set_param` can be called multiple times. Previously passed arguments are overwritten by successive calls.

        The following parameters are available:

        Args:
            overwrite(bool): Overwrites existing samplingplan under the same name, if set to ``True``.
            id_precision(str): Padding for IDs of created samples. Defaults to 3. This means sample 20 will be denoted as 020.
        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for SamplingPlanner.'.format(key))
            else:
                setattr(self, key, value)

    def set_sampling_var(self, name:str, fun_var_pdf:Callable[[],Union[float,int]]=None)->None:
        """Introduce new sampling variables to the :py:class:`SamplingPlanner`. Define variable name.
        Optionally add a function to generate values for the sampled variable (e.g. following some distribution).
        The parameter ``fun_var_pdf`` defaults to ``None``.

        Note:
            If no value-generating function is passed (for any of the introduced variables),
            all sampling cases must be created manually with :py:meth:`add_sampling_case`.

        Note:
            Value generating function ``fun_var_pdf`` must not require inputs.

        **Example:**

        ::

            sp = do_mpc.sampling.SamplingPlanner()

            # Plan with two variables alpha and beta:
            sp.set_sampling_var('alpha', np.random.randn)
            sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

        In the example we have passed a ``BuiltinFunction`` for the introduced variable ``alpha``.
        We use the function that created values from the random normal distribution with zero mean and unity covariance.
        For the variable ``beta`` we created a new lambda function that draws random integers from 0 to 5.

        Args:
            name: Name of the sampled variable
            fun_var_pdf: Declare the value-generating function of the sampled variable

        Raises:
            assertion: ``name`` must be string
            assertion: ``fun_var_pdf`` must be Function or BuiltinFunction
        """
        assert isinstance(name, str), 'name must be str, you have {}'.format(type(name))
        assert isinstance(fun_var_pdf, (types.FunctionType, types.BuiltinFunctionType, type(None))), 'fun_var_pdf must be either Function or BuiltinFunction_or_Method or None, you have {}'.format(type(fun_var_pdf))
        self.sampling_vars.append({'name':name, 'fun_var_pdf':fun_var_pdf})
        self.sampling_var_names.append(name)

    def add_sampling_case(self, **kwargs)->list:
        """ Manually add sampling case with user-defined values.
        Create a sampling case by choosing values for the previously introduced sampling variables (with :py:meth:`set_sampling_var`).

        Method takes arbitrary (keyword, argument) pairs, where the keywords must refer to previously introduced sampling variables.
        :py:meth:`add_sampling_case` will automatically augment the sampling case with values for variables that are not passed as arguments.
        This only works if these variables were created with the argument ``fun_var_pdf``.

        **Example:**

        ::

            sp = do_mpc.sampling.SamplingPlanner()

            # Plan with two variables alpha and beta:
            sp.set_sampling_var('alpha', np.random.randn)
            sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

            # Create two new sampling cases, missing variable is auto-generated:
            sp.add_sampling_case(alpha=1)
            sp.add_sampling_case(beta= 0)
        
        Returns:
            Returns the newly created sampling plan.
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
                # Augmentation is not possible if value generating function is not supplied.
                assert var['fun_var_pdf'] is not None, 'Cannot augment sampling_case for missing variable {}. Variable generating function is missing.'.format(var['name'])
                temp_dic[var['name']] = var['fun_var_pdf']()

        # Generate string ID of sampling case based on index and pad with zeros:
        id = len(self.sampling_plan)
        temp_dic['id'] = str(id).zfill(self.id_precision)

        self.sampling_plan.append(temp_dic)

        return self.sampling_plan



    def gen_sampling_plan(self, n_samples:int)->list:
        """Generate the sampling plan. The generated plan contains ``n_samples`` samples based on the defined variables and the corresponding evaluation functions.

        Args:
            n_samples: The number of generated samples

        Raises:
            assertion: n_samples must be int

        Returns:
            Returns the newly created sampling plan.
        """
        assert isinstance(n_samples, int), 'n_samples must be int, you have {}'.format(type(n_samples))
        assert n_samples>0, 'n_samples must be larger than 0.'

        for i in range(n_samples):
            self.add_sampling_case()


        return self.sampling_plan

    def product(self, **kwargs:dict)->list:
        """Cartesian product of input variables.
        This method is inspired by `itertools.product <https://docs.python.org/3/library/itertools.html#itertools.product>`_.

        Must pass a list for each ``sampling_var`` that should be considered. Not all ``sampling_vars`` must be referenced. 
        Sampling vars that are excluded, will generate a value according to their assigned ``fun_var_pdf`` (see :py:meth:`set_sampling_var`).

        Args:
            kwargs: Keyword arguments of the form ``var_name=var_values``.

        Returns:
            Returns the newly created sampling plan.
        """
        # Check if all key word values are lists:
        check = np.alltrue([isinstance(v, list) for v in kwargs.values()])
        if not check:
            raise ValueError('keyword values must be lists')

        # Check if all key words are existing sampling variables:
        keys = kwargs.keys()
        check = np.alltrue([v in self.sampling_var_names for v in keys])
        if not check:
            raise ValueError('keyword names must be existing sampling variables')

        # Create cartesian product of all values passen in kwargs:
        values = list(itertools.product(*list(kwargs.values())))

        # Create new sampling cases:
        for value in values:
            # Zip together the value(s) of the current case with the respective keys:
            case = dict(zip(keys, value))
            # Add sampling case
            self.add_sampling_case(**case)

        return self.sampling_plan
        

    def export(self, sampling_plan_name:str)->None:
        """Export SamplingPlan in pickle format.
        Pass ``sampling_plan_name`` without any path. File extension can be added (but will be stripped automatically).
        Change the path with :py:attr:`data_dir`.

        Args:
            sampling_plan_name: Name of the exported sampling plan file.

        Raises:
            assertion: ``sampling_plan_name`` must be string.
        """
        assert isinstance(sampling_plan_name, str), 'sampling_plan_name must be of type str. You have {}.'.format(type(sampling_plan_name))

        # Strip file extension from name:
        sampling_plan_name = os.path.splitext(sampling_plan_name)[0]

        full_name = self.data_dir + sampling_plan_name + '.pkl'
        if not os.path.isfile(full_name) or self.overwrite:
            save_pickle(full_name, self.sampling_plan)
        else:
            for i in range(1,10000):
                full_name = self.data_dir + sampling_plan_name + str(i) + '.pkl'
                if not os.path.isfile(full_name):
                    save_pickle(full_name, self.sampling_plan)
                    break