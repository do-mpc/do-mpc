#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import itertools
import do_mpc.data
from do_mpc import backend_optimizer
import time


class optimizer(backend_optimizer):
    """This is where the magic happens. The optimizer class is used to configure, setup and solve the MPC optimization problem
    for robust optimal control. The ocp is constructed based on the supplied instance of :py:class:`do_mpc.model`, which must be configured beforehand.

    **Configuration and setup:**

    Configuring and setting up the optimizer involves the following steps:

    1. Use :py:func:`optimizer.set_param` to configure the :py:class:`optimizer`. See docstring for details.

    2. Use :py:meth:`do_mpc.model.get_variables` to obtain the variables defined in :py:class:`do_mpc.model` and express the objective of the optimization problem in terms of these variables.

    3. Set the objective of the control problem with :py:func:`optimizer.set_objective`.

    4. Use :py:func:`optimizer.get_rterm` to obtain the structure of weighting parameters to penalize changes in the input and set appropriate values.  See docstring for details.

    5. Set upper and lower bounds.

    6. Optionally, set further (non-linear) constraints with :py:func:`optimizer.set_nl_cons`. See docstring for details.

    7. Use the low-level API (:py:func:`optimizer.get_p_template` and :py:func:`optimizer.set_p_fun`) or high level API (:py:func:`optimizer.set_uncertainty_values`) to create scenarios for robust MPC. See docstrings for details.

    8. Finally, call :py:func:`optimizer.setup`.

    """
    def __init__(self, model):
        super().__init__()

        self.model = model

        assert model.flags['setup'] == True, 'Model for optimizer was not setup. After the complete model creation call model.setup_model().'

        self.data = do_mpc.data.optimizer_data(self.model)

        self._x_lb = model._x(-np.inf)
        self._x_ub = model._x(np.inf)

        self._x_terminal_lb = model._x(-np.inf)
        self._x_terminal_ub = model._x(np.inf)

        self._u_lb = model._u(-np.inf)
        self._u_ub = model._u(np.inf)

        self._x_scaling = model._x(1)
        self._u_scaling = model._u(1)
        self._z_scaling = model._z(1)

        self.rterm_factor = self.model._u(0)

        # Lists for further non-linear constraints (optional). Constraints are formulated as lb < cons < 0
        self.nl_cons_list = [
            {'expr_name': 'default', 'expr': DM(), 'lb': DM()}
        ]

        self._x0 = model._x(0)
        self._u0 = model._u(0)
        self._z0 = model._z(0)
        self._t0 = np.array([0])

        # Parameters that can be set for the optimizer:
        self.data_fields = [
            'n_horizon',
            'n_robust',
            'open_loop',
            't_step',
            'state_discretization',
            'collocation_type',
            'collocation_deg',
            'collocation_ni',
            'store_full_solution',
            'store_lagr_multiplier',
            'store_solver_stats',
            'nlpsol_opts'
        ]

        # Default Parameters:
        self.n_robust = 0
        self.state_discretization = 'collocation'
        self.collocation_type = 'radau'
        self.collocation_deg = 2
        self.collocation_ni = 1
        self.open_loop = False
        self.store_full_solution = False
        self.store_lagr_multiplier = True
        self.store_solver_stats = [
            'success',
            't_wall_S',
            't_wall_S',
        ]
        self.nlpsol_opts = {} # Will update default options with this dict.

        self.flags = {
            'setup': False,
            'set_objective': False,
            'set_rterm': False,
            'set_tvp_fun': False,
            'set_p_fun': False,

        }

    def set_initial_state(self, x0, reset_history=False, set_intial_guess=True):
        """Set the intial state of the optimizer.
        Optionally resets the history. The history is empty upon creation of the optimizer.

        Optionally update the initial guess. The initial guess is first created with the .setup() method
        and uses the class attributes _x0, _u0, _z0 for all time instances, collocation points (if applicable)
        and scenarios (if applicable). If these values were net explicitly set by the user, they default to all zeros.


        :param x0: Initial state
        :type x0: numpy array
        :param reset_history: Resets the history of the optimizer, defaults to False
        :type reset_history: bool (,optional)
        :param set_intial_guess: Setting the initial state also updates the intial guess for the optimizer.
        :type set_intial_guess: bool (,optional)

        :return: None
        :rtype: None
        """
        assert x0.size == self.model._x.size, 'Intial state cannot be set because the supplied vector has the wrong size. You have {} and the model is setup for {}'.format(x0.size, self.model._x.size)
        assert isinstance(reset_history, bool), 'reset_history parameter must be of type bool. You have {}'.format(type(reset_history))
        if isinstance(x0, (np.ndarray, casadi.DM)):
            self._x0 = self.model._x(x0)
        elif isinstance(x0, structure3.DMStruct):
            self._x0 = x0
        else:
            raise Exception('x0 must be of tpye (np.ndarray, casadi.DM, structure3.DMStruct). You have: {}'.format(type(x0)))
        if reset_history:
            self.reset_history()

        if set_intial_guess:
            self.set_initial_guess()

    def reset_history(self):
        """Reset the history of the optimizer
        """
        self.data.init_storage()
        self._t0 = np.array([0])

    def set_param(self, **kwargs):
        """Method to set the parameters of the optimizer class. Parameters must be passed as pairs of valid keywords and respective argument.
        For example:
        ::
            optimizer.set_param(n_horizon = 20)

        It is also possible and convenient to pass a dictionary with multiple parameters simultaneously as shown in the following example:
        ::
            setup_optimizer = {
                'n_horizon': 20,
                't_step': 0.5,
            }
            optimizer.set_param(**setup_optimizer)

        .. note:: :py:func:`optimizer.set_param` can be called multiple times. Previously passed arguments are overwritten by successive calls.

        The following parameters are available:

        :param n_horizon: Prediction horizon of the optimal control problem. Parameter must be set by user.
        :type n_horizon: int

        :param n_robust: Robust horizon for robust scenario-tree MPC, defaults to ``0``. Optimization problem grows exponentially with ``n_robust``.
        :type n_robust: int , optional

        :param open_loop: Setting for scenario-tree MPC: If the parameter is ``False``, for each timestep **AND** scenario an individual control input is computed. If set to ``True``, the same control input is used for each scenario. Defaults to False.
        :type open_loop: bool , optional

        :param t_step: Timestep of the optimizer.
        :type t_step: float

        :param state_discretization: Choose the state discretization for continuous models. Currently only ``'collocation'`` is available. Defaults to ``'collocation'``.
        :type state_discretization: str

        :param collocation_type: Choose the collocation type for continuous models with collocation as state discretization. Currently only ``'radau'`` is available. Defaults to ``'radau'``.
        :type collocation_type: str

        :param collocation_deg: Choose the collocation degree for continuous models with collocation as state discretization. Defaults to ``2``.
        :type collocation_deg: int

        :param collocation_ni: Choose the collocation ni for continuous models with collocation as state discretization. Defaults to ``1``.
        :type collocation_ni: int

        :param store_full_solution: Choose whether to store the full solution of the optimization problem. This is required for animating the predictions in post processing. However, it drastically increases the required storage. Defaults to False.
        :type store_full_solution: bool

        :param store_lagr_multiplier: Choose whether to store the lagrange multipliers of the optimization problem. Increases the required storage. Defaults to ``True``.
        :type store_lagr_multiplier: bool

        :param store_solver_stats: Choose which solver statistics to store. Must be a list of valid statistics. Defaults to ``['success','t_wall_S','t_wall_S']``.
        :type store_solver_stats: list

        :param nlpsol_opts: Dictionary with options for the CasADi solver call ``nlpsol`` with plugin ``ipopt``. All options are listed `here <http://casadi.sourceforge.net/api/internal/d4/d89/group__nlpsol.html>`_.
        :type store_solver_stats: dict

        .. note:: We highly suggest to change the linear solver for IPOPT from `mumps` to `MA27`. In many cases this will drastically boost the speed of **do mpc**. Change the linear solver with:
            ::
                optimizer.set_param(nlpsol_opts = {'ipopt.linear_solver': 'MA27'})
        .. note:: To surpress the output of IPOPT, please use:
            ::
                surpress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
                optimizer.set_param(nlpsol_opts = surpress_ipopt)
        """
        assert self.flags['setup'] == False, 'Setting parameters after setup is prohibited.'

        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for optimizer.'.format(key))
            setattr(self, key, value)

    def set_nl_cons(self, expr_name, expr, lb=-np.inf):
        """Introduce new constraint to the optimizer class. Further constraints are optional.
        Expressions must be formulated with respect to _x, _u, _z, _tvp, _p.

        :param expr_name: Arbitrary name for the given expression. Names are used for key word indexing.
        :type expr_name: string
        :param expr: CasADi SX or MX function depending on _x, _u, _z, _tvp, _p.
        :type expr: CasADi SX or MX

        :raises assertion: expr_name must be str
        :raises assertion: expr must be a casadi SX or MX type

        :return: Returns the newly created expression. Expression can be used e.g. for the RHS.
        :rtype: casadi.SX
        """
        assert self.flags['setup'] == False, 'Cannot call .set_expression after .setup_model.'
        assert isinstance(expr_name, str), 'expr_name must be str, you have: {}'.format(type(expr_name))
        assert isinstance(expr, (casadi.SX, casadi.MX)), 'expr must be a casadi SX or MX type, you have: {}'.format(type(expr))
        assert isinstance(lb, (int, float, np.ndarray)), 'lb must be float, int or numpy.ndarray, you have: {}'.format(type(lb))

        self.nl_cons_list.extend([{'expr_name': expr_name, 'expr': expr, 'lb' : lb}])

        return expr


    def set_objective(self, mterm=None, lterm=None):
        """Sets the objective of the optimal control problem (OCP). We introduce the following notation:

        .. math::

           \min_{x,u,z}\quad \sum_{k=0}^{n-1} ( l(x_k,u_k,z_k,p) + \Delta u_k^T R \Delta u_k ) + m(x_n)

        :py:func:`optimizer.set_objective` is used to set the :math:`l(x_k,u_k,z_k,p)` (``lterm``) and :math:`m(x_N)` (``lterm``), where ``N`` is the prediction horizon.
        Please see :py:func:`optimizer.set_rterm` for the ``rterm``.

        :param lterm: Stage cost - **scalar** symbolic expression with respect to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``
        :type lterm:  CasADi SX or MX
        :param mterm: Terminal cost - **scalar** symbolic expression with respect to ``_x``
        :type mterm: CasADi SX or MX

        :raises assertion: mterm must have shape=(1,1) (scalar expression)
        :raises assertion: lterm must have shape=(1,1) (scalar expression)

        :return: None
        :rtype: None
        """
        assert mterm.shape == (1,1), 'mterm must have shape=(1,1). You have {}'.format(mterm.shape)
        assert lterm.shape == (1,1), 'lterm must have shape=(1,1). You have {}'.format(lterm.shape)
        assert self.flags['setup'] == False, 'Cannot call .set_objective after .setup_model.'

        self.flags['set_objective'] = True
        # TODO: Add docstring
        _x, _u, _z, _tvp, _p, _aux = self.model.get_variables()

        # TODO: Check if this is only a function of x
        self.mterm = mterm
        # TODO: This function should be evaluated with scaled variables.
        self.mterm_fun = Function('mterm', [_x], [mterm])

        self.lterm = lterm
        self.lterm_fun = Function('lterm', [_x, _u, _z, _tvp, _p], [lterm])

    def set_rterm(self, **kwargs):
        """Set the penality factor for the inputs. Call this function with keyword argument refering to the input names in
        :py:class:`model` and the penalty factor as the respective value.

        Example:
        ::
            # in model definition:
            Q_heat = model.set_variable(var_type='_u', var_name='Q_heat')
            F_flow = model.set_variable(var_type='_u', var_name='F_flow')

            ...
            # in optimizer configuration:
            optimizer.set_rterm(Q_heat = 10)
            optimizer.set_rterm(F_flow = 10)
            # or alternatively:
            optimizer.set_rterm(Q_heat = 10, F_flow = 10)
        """
        assert self.flags['setup'] == False, 'Cannot call .set_rterm after .setup_model.'

        self.flags['set_rterm'] = True
        for key, val in kwargs.items():
            assert key in self.model._u.keys(), 'Must pass keywords that refer to input names defined in model. Valid is: {}. You have: {}'.format(self.model._u.keys(), key)
            assert isinstance(val, (int, float, np.ndarray)), 'Value for {} must be int, float or numpy.ndarray. You have: {}'.format(key, type(val))
            self.rterm_factor[key] = val


    def get_tvp_template(self):
        """The method returns a structured object with n_horizon elements, and a set of time varying parameters (as defined in model)
        for each of these instances. The structure is initialized with all zeros. Use this object to define values of the time varying parameters.

        This structure (with numerical values) should be used as the output of the tvp_fun function which is set to the class with .set_tvp_fun (see doc string).
        Use the combination of .get_tvp_template() and .set_tvp_fun().

        Example:
        ::
            # in model definition:
            alpha = model.set_variable(var_type='_tvp', var_name='alpha')
            beta = model.set_variable(var_type='_tvp', var_name='beta')

            ...
            # in optimizer configuration:
            (assume n_horizon = 5)
            tvp_temp_1 = optimizer.get_tvp_template()
            tvp_temp_1['_tvp', :] = np.array([1,1])

            tvp_temp_2 = optimizer.get_tvp_template()
            tvp_temp_2['_tvp', :] = np.array([0,0])

            def tvp_fun(t_now):
                if t_now<10:
                    return tvp_temp_1
                else:
                    tvp_temp_2

            optimizer.set_tvp_fun(tvp_fun)

        :return: None
        :rtype: None
        """

        tvp_template = struct_symSX([
            entry('_tvp', repeat=self.n_horizon, struct=self.model._tvp)
        ])
        return tvp_template(0)

    def set_tvp_fun(self, tvp_fun):
        """ Set the tvp_fun which is called at each optimization step to get the current prediction of the time-varying parameters.
        The supplied function must be callable with the current time as the only input. Furthermore, the function must return
        a CasADi structured object which is based on the horizon and on the model definition. The structure can be obtained with
        .get_tvp_template().
        ::
            # in model definition:
            alpha = model.set_variable(var_type='_tvp', var_name='alpha')
            beta = model.set_variable(var_type='_tvp', var_name='beta')

            ...
            # in optimizer configuration:
            (assume n_horizon = 5)
            tvp_temp_1 = optimizer.get_tvp_template()
            tvp_temp_1['_tvp', :] = np.array([1,1])

            tvp_temp_2 = optimizer.get_tvp_template()
            tvp_temp_2['_tvp', :] = np.array([0,0])

            def tvp_fun(t_now):
                if t_now<10:
                    return tvp_temp_1
                else:
                    tvp_temp_2

            optimizer.set_tvp_fun(tvp_fun)

        The method .set_tvp_fun() must be called prior to setup IF time-varying parameters are defined in the model.

        :param tvp_fun: Function that returns the predicted tvp values at each timestep. Must have single input (float) and return a structure3.DMStruct (obtained with .get_tvp_template())
        :type tvp_fun: function

        """
        assert isinstance(tvp_fun(0), structure3.DMStruct), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'
        assert self.get_tvp_template().labels() == tvp_fun(0).labels(), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'

        self.flags['set_tvp_fun'] = True

        self.tvp_fun = tvp_fun

    def get_p_template(self, n_combinations):
        """ Low level API method to set user defined scenarios for robust MPC but defining an arbitrary number
        of combinations for the parameters defined in the model. The method returns a structured object which is
        initialized with all zeros. Use this object to define value of the parameters for an arbitrary number of scenarios (defined by n_scenarios).

        This structure (with numerical values) should be used as the output of the p_fun function which is set to the class with .set_p_fun (see doc string).

        Use the combination of .get_p_template() and .set_p_template() as a more adaptable alternative to .set_uncertainty_values().

        Example:
        ::
            # in model definition:
            alpha = model.set_variable(var_type='_p', var_name='alpha')
            beta = model.set_variable(var_type='_p', var_name='beta')

            ...
            # in optimizer configuration:
            n_combinations = 3
            p_template = optimizer.get_p_template(n_combinations)
            p_template['_p',0] = np.array([1,1])
            p_template['_p',1] = np.array([0.9, 1.1])
            p_template['_p',2] = np.array([1.1, 0.9])

            def p_fun(t_now):
                return p_template

            optimizer.set_p_fun(p_fun)

        Note the nominal case is now:
        alpha = 1
        beta = 1
        which is determined by the order in the arrays above (first element is nominal).

        :param n_combinations: Define the number of combinations for the uncertain parameters for robust MPC.
        :type n_combinations: int

        :return: None
        :rtype: None
        """
        self.n_combinations = n_combinations
        p_template = struct_symSX([
            entry('_p', repeat=n_combinations, struct=self.model._p)
        ])
        return p_template(0)

    def set_p_fun(self, p_fun):
        """ Low level API method to set user defined scenarios for robust MPC but defining an arbitrary number
        of combinations for the parameters defined in the model. The method takes as input a function, which MUST
        return a structured object, based on the defined parameters and the number of combinations.
        The defined function has time as a single input.

        Obtain this structured object first, by calling :py:func:`optimizer.get_p_template`.

        Use the combination of :py:func:`optimizer.get_p_template` and :py:func:`optimizer.set_p_fun` as a more adaptable alternative to :py:func:`optimizer.set_uncertainty_values`.

        Example:
        ::
            # in model definition:
            alpha = model.set_variable(var_type='_p', var_name='alpha')
            beta = model.set_variable(var_type='_p', var_name='beta')

            ...
            # in optimizer configuration:
            n_combinations = 3
            p_template = optimizer.get_p_template(n_combinations)
            p_template['_p',0] = np.array([1,1])
            p_template['_p',1] = np.array([0.9, 1.1])
            p_template['_p',2] = np.array([1.1, 0.9])

            def p_fun(t_now):
                return p_template

            optimizer.set_p_fun(p_fun)

        Note the nominal case is now:
        alpha = 1
        beta = 1
        which is determined by the order in the arrays above (first element is nominal).

        :param p_fun: Function which returns a structure with numerical values. Must be the same structure as obtained from :py:func:`optimizer.get_p_template`.
        Function must have a single input (time).
        :type p_fun: function

        :return: None
        :rtype: None
        """
        assert self.get_p_template(self.n_combinations).labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.flags['set_p_fun'] = True
        self.p_fun = p_fun

    def set_uncertainty_values(self, uncertainty_values):
        """ High-level API method to conveniently set all possible scenarios for multistage MPC, given a list of uncertainty values.
        This list must have the same number of elements as uncertain parameters in the model definition. The first element is the nominal case.
        Each list element can be an array or list of possible values for the respective parameter.
        Note that the order of elements determine the assignment.

        Example:
        ::
            # in model definition:
            alpha = model.set_variable(var_type='_p', var_name='alpha')
            beta = model.set_variable(var_type='_p', var_name='beta')
            ...
            # in optimizer configuration:
            alpha_var = np.array([1., 0.9, 1.1])
            beta_var = np.array([1., 1.05])
            optimizer.set_uncertainty_values([alpha_var, beta_var])

        Note the nominal case is now:
        alpha = 1
        beta = 1
        which is determined by the order in the arrays above (first element is nominal).

        :param uncertainty_values: List of lists / numpy arrays with the same number of elements as number of parameters in model.
        :type uncertainty_values: list

        :raises asssertion: uncertainty values must be of type list

        :return: None
        :rtype: None
        """
        assert isinstance(uncertainty_values, list), 'uncertainty values must be of type list, you have: {}'.format(type(uncertainty_values))

        p_scenario = list(itertools.product(*uncertainty_values))
        n_combinations = len(p_scenario)
        p_template = self.get_p_template(n_combinations)
        p_template['_p', :] = p_scenario

        def p_fun(t_now):
            return p_template

        self.set_p_fun(p_fun)

    def check_validity(self):
        # Objective mus be defined.
        if self.flags['set_objective'] == False:
            raise Exception('Objective is undefined. Please call .set_objective() prior to .setup().')
        # rterm should have been set (throw warning if not)
        if self.flags['set_rterm'] == False:
            warning('rterm was not set and defaults to zero. Changes in the control inputs are not penalized. Can lead to oscillatory behavior.')
            time.sleep(2)
        # tvp_fun must be set, if tvp are defined in model.
        if self.flags['set_tvp_fun'] == False and self.model._tvp.size > 0:
            raise Exception('You have not supplied a function to obtain the time varying parameters defined in model. Use .set_tvp_fun() prior to setup.')
        # p_fun must be set, if p are defined in model.
        if self.flags['set_p_fun'] == False and self.model._p.size > 0:
            raise Exception('You have not supplied a function to obtain the parameters defined in model. Use .set_p_fun() (low-level API) or .set_uncertainty_values() (high-level API) prior to setup.')

        if np.any(self.rterm_factor.cat.full() < 0):
            warning('You have selected negative values for the rterm penalizing changes in the control input.')
            time.sleep(2)

        # Set dummy functions for tvp and p in case these parameters are unused.
        if 'tvp_fun' not in self.__dict__:
            _tvp = self.get_tvp_template()

            def tvp_fun(t): return _tvp
            self.set_tvp_fun(tvp_fun)

        if 'p_fun' not in self.__dict__:
            _p = self.get_p_template()

            def p_fun(t): return _p
            self.set_p_fun(p_fun)

    def setup(self):
        """The setup method finalizes the optimizer creation. After this call, the .solve() method is applicable.
        The method wraps the following calls:

        * check_validity

        * setup_nlp

        * set_inital_guess

        * prepare_data

        and sets the setup flag = True.

        """
        self.flags['setup'] = True

        # Create struct for _nl_cons:
        # Use the previously defined SX.sym variables to declare shape and symbolic variable.
        self._nl_cons = struct_SX([
            entry(expr_i['expr_name'], expr=expr_i['expr']) for expr_i in self.nl_cons_list
        ])
        # Make function from these expressions:
        _x, _u, _z, _tvp, _p, _aux = self.model.get_variables()
        self._nl_cons_fun = Function('nl_cons_fun', [_x, _u, _z, _tvp, _p], [self._nl_cons])
        # Create bounds:
        self._nl_cons_ub = self._nl_cons(0)
        self._nl_cons_lb = self._nl_cons(-np.inf)
        # Set bounds:
        for nl_cons_i in self.nl_cons_list:
            self._nl_cons_lb[nl_cons_i['expr_name']] = nl_cons_i['lb']

        self.check_validity()
        self.setup_nlp()
        self.set_initial_guess()
        self.prepare_data()

    def prepare_data(self):
        """Write optimizer meta data to data object (all params set in self.data_fields).
        If selected, initialize the container for the full solution of the optimizer.
        """
        meta_data = {key: getattr(self, key) for key in self.data_fields}
        meta_data.update({'structure_scenario': self.scenario_tree['structure_scenario']})
        self.data.set_meta(**meta_data)

        if self.store_full_solution == True:
            # Create data_field for the optimal solution.
            self.data.data_fields.update({'_opt_x_num': self.n_opt_x})
            self.data.data_fields.update({'_opt_aux_num': self.n_opt_aux})
            self.data.opt_x = self.opt_x
            # aux_struct is the struct_symSX variant of opt_aux (which is struct_SX). struct_SX cannot be unpickled (bug).
            # See: https://groups.google.com/forum/#!topic/casadi-users/dqAb4tnA2ik
            self.data.opt_aux = self.aux_struct
        if self.store_lagr_multiplier == True:
            # Create data_field for the lagrange multipliers
            self.data.data_fields.update({'_lam_g_num': self.n_opt_lagr})
        if len(self.store_solver_stats) > 0:
            # These are valid arguments for solver stats:
            solver_stats = ['iter_count', 'iterations', 'n_call_S', 'n_call_callback_fun',
                            'n_call_nlp_f', 'n_call_nlp_g', 'n_call_nlp_grad', 'n_call_nlp_grad_f',
                            'n_call_nlp_hess_l', 'n_call_nlp_jac_g', 'return_status', 'success', 't_proc_S',
                            't_proc_callback_fun', 't_proc_nlp_f', 't_proc_nlp_g', 't_proc_nlp_grad',
                            't_proc_nlp_grad_f', 't_proc_nlp_hess_l', 't_proc_nlp_jac_g', 't_wall_S',
                            't_wall_callback_fun', 't_wall_nlp_f', 't_wall_nlp_g', 't_wall_nlp_grad', 't_wall_nlp_grad_f',
                            't_wall_nlp_hess_l', 't_wall_nlp_jac_g']
            # Create data_field(s) for the recorded (valid) stats.
            for stat_i in self.store_solver_stats:
                assert stat_i in solver_stats, 'The requested {} is not a valid solver stat and cannot be recorded. Please supply one of the following (or none): {}'.format(stat_i, solver_stats)
                self.data.data_fields.update({stat_i: 1})

        self.data.init_storage()

    def set_initial_guess(self):
        """Uses the current class attributes _x0, _z0 and _u0 to create an initial guess for the optimizer.
        The initial guess is simply the initial values for all instances of x, u and z. The method is automatically
        evoked when calling the .setup() method.
        However, if no initial values for x, u and z were supplied during setup, these default to zero.
        """
        assert self.flags['setup'] == True, 'optimizer was not setup yet. Please call optimizer.setup().'

        self.opt_x_num['_x'] = self._x0.cat/self._x_scaling
        self.opt_x_num['_u'] = self._u0.cat/self._u_scaling
        self.opt_x_num['_z'] = self._z0.cat/self._z_scaling

    def solve(self):
        """Solves the optmization problem. The current time-step is defined by the parameters in the
        self.opt_p_num CasADi structured Data. These include the initial condition, the parameters, the time-varying paramters and the previous u.
        Typically, self.opt_p_num is prepared for the current iteration in the configuration.make_step_optimizer() method.
        It is, however, valid and possible to directly set paramters in self.opt_p_num before calling .solve().

        Solve updates the opt_x_num, and lam_g_num attributes of the class. In resetting, opt_x_num to the current solution, the method implicitly
        enables warmstarting the optimizer for the next iteration, since this vector is always used as the initial guess.

        :raises asssertion: optimizer was not setup yet.

        :return: None
        :rtype: None
        """
        assert self.flags['setup'] == True, 'optimizer was not setup yet. Please call optimizer.setup().'

        r = self.S(x0=self.opt_x_num, lbx=self.lb_opt_x, ubx=self.ub_opt_x,  ubg=self.cons_ub, lbg=self.cons_lb, p=self.opt_p_num)
        # Note: .master accesses the underlying vector of the structure.
        self.opt_x_num.master = r['x']
        self.opt_x_num_unscaled.master = r['x']*self.opt_x_scaling
        self.opt_g_num = r['g']
        # Values of lagrange multipliers:
        self.lam_g_num = r['lam_g']
        self.solver_stats = self.S.stats()

        # Calculate values of auxiliary expressions (defined in model)
        self.opt_aux_num.master = self.opt_aux_expression_fun(
                self.opt_x_num,
                self.opt_p_num
            )



    def make_step(self, x0):
        """Main method of the optimizer class during control runtime. This method is called at each timestep
        and returns the control input for the current initial state ``x0``.

        The method prepares the optimizer by setting the current parameters, calls :py:func:`optimizer.solve`
        and updates the :py:class:`do_mpc.data` object.

        :param x0: Current state of the system.
        :type x0: numpy.ndarray

        :return: u0
        :rtype: numpy.ndarray
        """

        u_prev = self._u0
        tvp0 = self.tvp_fun(self._t0)
        p0 = self.p_fun(self._t0)
        t0 = self._t0

        self.opt_p_num['_x0'] = x0
        self.opt_p_num['_u_prev'] = u_prev
        self.opt_p_num['_tvp'] = tvp0['_tvp']
        self.opt_p_num['_p'] = p0['_p']
        self.solve()

        u0 = self._u0 = self.opt_x_num['_u', 0, 0]*self._u_scaling
        z0 = self._z0 = self.opt_x_num['_z', 0, 0, -1]*self._z_scaling
        aux0 = self.opt_aux_num['_aux', 0, 0]

        self.data.update(_x = x0)
        self.data.update(_u = u0)
        self.data.update(_z = z0)
        #TODO: tvp und p support.
        # self.data.update(_tvp = tvp0)
        # self.data.update(_p = p0)
        self.data.update(_time = t0)
        self.data.update(_aux_expression = aux0)

        # Store additional information
        if self.store_full_solution == True:
            opt_x_num_unscaled = self.opt_x_num_unscaled
            opt_aux_num = self.opt_aux_num
            self.data.update(_opt_x_num = opt_x_num_unscaled)
            self.data.update(_opt_aux_num = opt_aux_num)
        if self.store_lagr_multiplier == True:
            lam_g_num = self.lam_g_num
            self.data.update(_lam_g_num = lam_g_num)
        if len(self.store_solver_stats) > 0:
            solver_stats = self.solver_stats
            store_solver_stats = self.store_solver_stats
            self.data.update(**{stat_i: value for stat_i, value in solver_stats.items() if stat_i in store_solver_stats})

        self._t0 = self._t0 + self.t_step

        return u0.full()
