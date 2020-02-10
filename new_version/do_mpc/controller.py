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
from indexedproperty import IndexedProperty
import time

import do_mpc.data
import do_mpc.optimizer

class MPC(do_mpc.optimizer.Optimizer):
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

        self.model = model
        assert model.flags['setup'] == True, 'Model for optimizer was not setup. After the complete model creation call model.setup_model().'
        self.data = do_mpc.data.Data(self.model)
        self.data.dtype = 'MPC'
        # Initialize structure for intial conditions:
        self._x0 = model._x(0.0)
        self._u0 = model._u(0.0)
        self._z0 = model._z(0.0)
        self._t0 = np.array([0.0])

        # Initialize parent class:
        do_mpc.optimizer.Optimizer.__init__(self, model)

        # Initialize further structures specific to the MPC optimization problem.
        # This returns an identical numerical structure with all values set to the passed value.
        self._x_terminal_lb = model._x(-np.inf)
        self._x_terminal_ub = model._x(np.inf)

        self.rterm_factor = self.model._u(0.0)

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

        # Flags are checked when calling .setup_optimizer.
        self.flags = {
            'setup': False,
            'set_objective': False,
            'set_rterm': False,
            'set_tvp_fun': False,
            'set_p_fun': False,
        }


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
            else:
                setattr(self, key, value)


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
        _x, _u, _z, _tvp, _p, _aux,  *_ = self.model.get_variables()

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

        # Lower bounds should be lower than upper bounds:
        for lb, ub in zip([self._x_lb, self._u_lb, self._z_lb], [self._x_ub, self._u_ub, self._z_ub]):
            bound_check = lb.cat > ub.cat
            bound_fail = [label_i for i,label_i in enumerate(lb.labels()) if bound_check[i]]
            if np.any(bound_check):
                raise Exception('Your bounds are inconsistent. For {} you have lower bound > upper bound.'.format(bound_fail))

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
        _x, _u, _z, _tvp, _p, _aux, *_ = self.model.get_variables()
        self._nl_cons_fun = Function('nl_cons_fun', [_x, _u, _z, _tvp, _p], [self._nl_cons])
        # Create bounds:
        self._nl_cons_ub = self._nl_cons(0)
        self._nl_cons_lb = self._nl_cons(-np.inf)
        # Set bounds:
        for nl_cons_i in self.nl_cons_list:
            self._nl_cons_lb[nl_cons_i['expr_name']] = nl_cons_i['lb']


        self.check_validity()
        self._setup_mpc_optim_problem()
        self.set_initial_guess()

        # Gather meta information:
        meta_data = {key: getattr(self, key) for key in self.data_fields}
        meta_data.update({'structure_scenario': self.scenario_tree['structure_scenario']})
        self.data.set_meta(**meta_data)

        self.prepare_data()

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

        u0 = self.opt_x_num['_u', 0, 0]*self._u_scaling
        z0 = self.opt_x_num['_z', 0, 0, -1]*self._z_scaling
        aux0 = self.opt_aux_num['_aux', 0, 0]

        self.data.update(_x = x0)
        self.data.update(_u = u0)
        self.data.update(_z = z0)
        #TODO: tvp und p support.
        # self.data.update(_tvp = tvp0)
        #self.data.update(_p = p0)
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

        # Update initial
        self._t0 = self._t0 + self.t_step
        self._x0.master = x0
        self._u0.master = u0
        self._z0.master = z0

        return u0.full()


    def _setup_mpc_optim_problem(self):
        # Obtain an integrator (collocation, discrete-time) and the amount of intermediate (collocation) points
        ifcn, n_total_coll_points = self._setup_discretization()
        n_branches, n_scenarios, child_scenario, parent_scenario, branch_offset = self._setup_scenario_tree()
        n_max_scenarios = self.n_combinations ** self.n_robust
        # Create struct for optimization variables:
        self.opt_x = opt_x = struct_symSX([
            entry('_x', repeat=[self.n_horizon+1, n_max_scenarios,
                                1+n_total_coll_points], struct=self.model._x),
            entry('_z', repeat=[self.n_horizon, n_max_scenarios,
                                1+n_total_coll_points], struct=self.model._z),
            entry('_u', repeat=[self.n_horizon, n_max_scenarios], struct=self.model._u),
        ])
        self.n_opt_x = self.opt_x.shape[0]
        # NOTE: The entry _x[k,child_scenario[k,s,b],:] starts with the collocation points from s to b at time k
        #       and the last point contains the child node
        # NOTE: Currently there exist dummy collocation points for the initial state (for each branch)

        # Create scaling struct as assign values for _x, _u, _z.
        self.opt_x_scaling = opt_x_scaling = opt_x(1)
        opt_x_scaling['_x'] = self._x_scaling
        opt_x_scaling['_z'] = self._z_scaling
        opt_x_scaling['_u'] = self._u_scaling
        # opt_x are unphysical (scaled) variables. opt_x_unscaled are physical (unscaled) variables.
        self.opt_x_unscaled = opt_x_unscaled = opt_x(opt_x.cat * opt_x_scaling)


        # Create struct for optimization parameters:
        self.opt_p = opt_p = struct_symSX([
            entry('_x0', struct=self.model._x),
            entry('_tvp', repeat=self.n_horizon, struct=self.model._tvp),
            entry('_p', repeat=self.n_combinations, struct=self.model._p),
            entry('_u_prev', struct=self.model._u)
        ])

        # Dummy struct with symbolic variables
        self.aux_struct = struct_symSX([
            entry('_aux', repeat=[self.n_horizon, n_max_scenarios], struct=self.model._aux_expression)
        ])
        # Create mutable symbolic expression from the struct defined above.
        self.opt_aux = opt_aux = struct_SX(self.aux_struct)

        self.n_opt_aux = self.opt_aux.shape[0]

        self.lb_opt_x = opt_x(-np.inf)
        self.ub_opt_x = opt_x(np.inf)

        # Initialize objective function and constraints
        obj = 0
        cons = []
        cons_lb = []
        cons_ub = []

        # Initial condition:
        cons.append(opt_x['_x', 0, 0, -1]-opt_p['_x0']/self._x_scaling)

        cons_lb.append(np.zeros((self.model.n_x, 1)))
        cons_ub.append(np.zeros((self.model.n_x, 1)))

        # NOTE: Weigthing factors for the tree assumed equal. They could be set from outside
        # Weighting factor for every scenario
        omega = [1. / n_scenarios[k + 1] for k in range(self.n_horizon)]
        omega_delta_u = [1. / n_scenarios[k + 1] for k in range(self.n_horizon)]

        # For all control intervals
        for k in range(self.n_horizon):
            # For all scenarios (grows exponentially with n_robust)
            for s in range(n_scenarios[k]):
                # For all childen nodes of each node at stage k, discretize the model equations
                for b in range(n_branches[k]):
                    # Obtain the index of the parameter values that should be used for this scenario
                    current_scenario = b + branch_offset[k][s]

                    # Compute constraints and predicted next state of the discretization scheme
                    [g_ksb, xf_ksb] = ifcn(opt_x['_x', k, s, -1], vertcat(*opt_x['_x', k+1, child_scenario[k][s][b], :-1]),
                                           opt_x['_u', k, s], vertcat(*opt_x['_z', k, s, :]), opt_p['_tvp', k], opt_p['_p', current_scenario])

                    # Add the collocation equations
                    cons.append(g_ksb)
                    cons_lb.append(np.zeros(g_ksb.shape[0]))
                    cons_ub.append(np.zeros(g_ksb.shape[0]))

                    # Add continuity constraints
                    cons.append(xf_ksb - opt_x['_x', k+1, child_scenario[k][s][b], -1])
                    cons_lb.append(np.zeros((self.model.n_x, 1)))
                    cons_ub.append(np.zeros((self.model.n_x, 1)))

                    # Add nonlinear constraints only on each control step
                    nl_cons_k = self._nl_cons_fun(
                        opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s], opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])
                    cons.append(nl_cons_k)
                    cons_lb.append(self._nl_cons_lb)
                    cons_ub.append(self._nl_cons_ub)

                    # Add terminal constraints
                    # TODO: Add terminal constraints with an additional nl_cons

                    # Add contribution to the cost
                    obj += omega[k] * self.lterm_fun(opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s],
                                                     opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])
                    # In the last step add the terminal cost too
                    if k == self.n_horizon - 1:
                        obj += omega[k] * self.mterm_fun(opt_x_unscaled['_x', k + 1, s, -1])

                    # U regularization:
                    if k == 0:
                        obj += self.rterm_factor.cat.T@((opt_x['_u', 0, s]-opt_p['_u_prev']/self._u_scaling)**2)
                    else:
                        obj += self.rterm_factor.cat.T@((opt_x['_u', k, s]-opt_x['_u', k-1, parent_scenario[k][s]])**2)

                    # Calculate the auxiliary expressions for the current scenario:
                    opt_aux['_aux', k, s] = self.model._aux_expression_fun(
                        opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s], opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])

                # Bounds for the states on all discretize values along the horizon
                self.lb_opt_x['_x', k, s, :] = self._x_lb.cat/self._x_scaling
                self.ub_opt_x['_x', k, s, :] = self._x_ub.cat/self._x_scaling

                # Bounds for the inputs along the horizon
                self.lb_opt_x['_u', k, s] = self._u_lb.cat/self._u_scaling
                self.ub_opt_x['_u', k, s] = self._u_ub.cat/self._u_scaling

                # Bounds on the terminal state
                if k == self.n_horizon - 1:
                    self.lb_opt_x['_x', self.n_horizon, child_scenario[k][s][b], -1] = self._x_lb.cat/self._x_scaling
                    self.ub_opt_x['_x', self.n_horizon, child_scenario[k][s][b], -1] = self._x_ub.cat/self._x_scaling

        cons = vertcat(*cons)
        self.cons_lb = vertcat(*cons_lb)
        self.cons_ub = vertcat(*cons_ub)

        self.n_opt_lagr = cons.shape[0]
        # Create casadi optimization object:
        nlpsol_opts = {
            'expand': False,
            'ipopt.linear_solver': 'mumps',
        }.update(self.nlpsol_opts)
        nlp = {'x': vertcat(opt_x), 'f': obj, 'g': cons, 'p': vertcat(opt_p)}
        self.S = nlpsol('S', 'ipopt', nlp, self.nlpsol_opts)

        # Create copies of these structures with numerical values (all zero):
        self.opt_x_num = self.opt_x(0)
        self.opt_x_num_unscaled = self.opt_x(0)
        self.opt_p_num = self.opt_p(0)
        self.opt_aux_num = self.opt_aux(0)

        # Create function to caculate all auxiliary expressions:
        self.opt_aux_expression_fun = Function('opt_aux_expression_fun', [opt_x, opt_p], [opt_aux])
