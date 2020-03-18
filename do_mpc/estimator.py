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
import copy

import do_mpc.optimizer
import do_mpc.data


class Estimator:
    """The Estimator base class. Used for :py:class:`StateFeedback`, :py:class:`EKF` and :py:class:`MHE`.
    This class cannot be used independently.

    .. note::
       The methods :py:func:`Estimator.set_initial_state` and :py:func:`Estimator.reset_history`
       are overwritten when using the :py:class:`MHE` by the methods defined in :py:class:`do_mpc.optimizer.Optimizer`.

    """
    def __init__(self, model):
        self.model = model

        assert model.flags['setup'] == True, 'Model for estimator was not setup. After the complete model creation call model.setup_model().'

        self._x0 = model._x(0.0)
        self._u0 = model._u(0.0)
        self._z0 = model._z(0.0)
        self._t0 = np.array([0.0])

        self.data = do_mpc.data.Data(model)
        self.data.dtype = 'Estimator'


    def set_initial_state(self, x0, reset_history=False):
        """Set the intial state of the estimator.
        Optionally resets the history. The history is empty upon creation of the estimator.
        This method is overwritten for the :py:class:`MHE` from :py:class:`do_mpc.optimizer.Optimizer`.

        :param x0: Initial state
        :type x0: numpy array
        :param reset_history: Resets the history of the estimator, defaults to False
        :type reset_history: bool (,optional)

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

    def reset_history(self):
        """Reset the history of the estimator
        """
        self.data.init_storage()


class StateFeedback(Estimator):
    """Simple state-feedback "estimator".
    The main method :py:func:`StateFeedback.make_step` simply returns the input.
    Why do you even bother to use this class?
    """
    def __init__(self, model):
        super().__init__(model)

    def make_step(self, y0):
        """Return the measurement ``y0``.
        """
        return y0

class EKF(Estimator):
    """Extended Kalman Filter. Setup this class and use :py:func:`EKF.make_step`
    during runtime to obtain the currently estimated states given the measurements ``y0``.

    .. warning::
        Not currently implemented.
    """
    def __init__(self, model):
        raise Exception('EKF is not currently supported. This is a placeholder.')
        super().__init__(model)

        # Flags are checked when calling .setup.
        self.flags = {
            'setup': False,
        }

    def make_step(self, y0):
        """Main method during runtime. Pass the most recent measurement and
        retrieve the estimated state."""
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'
        None

class MHE(do_mpc.optimizer.Optimizer, Estimator):
    """Moving horizon estimator. THE MHE estimator extends the :py:class:`do_mpc.optimizer.Optimizer` base class
    (which is also used for the MPC controller), as well as the :py:class:`Estimator` base class.
    Use this class to configure and run the MHE based on a previously configured :py:class:`do_mpc.model.Model` instance.

    The class is initiated by passing a list of the **parameters that should be estimated**. This must be a subset (or all) of the parameters defined in
    :py:class:`do_mpc.model.Model`. This allows to define parameters in the model that influence the model externally (e.g. weather predictions),
    and those that are internal e.g. system parameters and can be estimated.
    Passing an empty list (default) value, means that no parameters are estimated.

    .. note::
        Parameters are influencing the model equation at all timesteps but are constant over the entire horizon.
        Parameters could also be introduced as states without dynamic but this would increase the total number of optimization variables.


    **Configuration and setup:**

    Configuring and setting up the MHE involves the following steps:

    1. Use :py:func:`MHE.set_param` to configure the :py:class:`MHE`. See docstring for details.

    2. Obtain the following variables from the class: ``MHE._y_meas``, ``MHE._y_calc``, ``MHE._x_prev``, ``MHE._x0``, ``MHE._p_est_prev``, ``MHE._p_est0``

    3. Set the objective of the control problem with :py:func:`MHE.set_objective` or use the high-level interface, :py:func:`MHE.set_default_objective`

    5. Set upper and lower bounds.

    6. Optionally, set further (non-linear) constraints with :py:func:`do_mpc.optimizer.Optimizer.set_nl_cons`.

    7. Use :py:func:`MHE.get_p_template` and :py:func:`MHE.set_p_fun` to set the function for the parameters.

    8. Finally, call :py:func:`MHE.setup`.

    During runtime use :py:func:`MHE.make_step` with the most recent measurement to obtain the estimated states.

    :param model: A configured and setup :py:class:`do_mpc.model.Model`
    :type model: :py:class:`do_mpc.model.Model`

    :param p_est_list: List with names of parameters (``_p``) defined in ``model``
    :type p_est_list: list

    """

    opt_x_num = None
    """Full MHE solution and initial guess.

    This is the core attribute of the MHE class.
    It is used as the initial guess when solving the optimization problem
    and then overwritten with the current solution.

    The attribute is a CasADi numeric structure with nested power indices.
    It can be indexed as follows:

    ::

        # dynamic states:
        opt_x_num['_x', time_step, collocation_point, _x_name]
        # algebraic states:
        opt_x_num['_z', time_step, collocation_point, _z_name]
        # inputs:
        opt_x_num['_u', time_step, _u_name]
        # estimated parameters:
        opt_x_Num['_p_est', _p_names]
        # slack variables for soft constraints:
        opt_x_num['_eps', time_step, _nl_cons_name]

    The names refer to those given in the :py:class:`do_mpc.model.Model` configuration.
    Further indices are possible, if the variables are itself vectors or matrices.

    The attribute can be used **to manually set a custom initial guess or for debugging purposes**.

    .. note::

        The attribute ``opt_x_num`` carries the scaled values of all variables. See ``opt_x_num_unscaled``
        for the unscaled values (these are not used as the initial guess).

    .. warning::

        Do not tweak or overwrite this attribute unless you known what you are doing.

    .. note::

        The attribute is populated when calling :py:func:`MHE.setup`
    """

    opt_p_num = None
    """Full MHE parameter vector.

    This attribute is used when calling the solver to pass all required parameters,
    including

    * previously estimated state(s)

    * previously estimated parameter(s)

    * known parameters

    * sequence of time-varying parameters

    * sequence of measurements parameters

    **do-mpc** handles setting these parameters automatically in the :py:func:`MHE.make_step`
    method. However, you can set these values manually and directly call :py:func:`MHE.solve`.

    The attribute is a CasADi numeric structure with nested power indices.
    It can be indexed as follows:

    ::

        # previously estimated state:
        opt_p_num['_x_prev', _x_name]
        # previously estimated parameters:
        opt_p_num['_p_est_prev', _x_name]
        # known parameters
        opt_p_num['_p_set', _p_name]
        # time-varying parameters:
        opt_p_num['_tvp', time_step, _tvp_name]
        # sequence of measurements:
        opt_p_num['_y_meas', time_step, _y_name]

    The names refer to those given in the :py:class:`do_mpc.model.Model` configuration.
    Further indices are possible, if the variables are itself vectors or matrices.

    .. warning::

        Do not tweak or overwrite this attribute unless you known what you are doing.

    .. note::

        The attribute is populated when calling :py:func:`MHE.setup`

    """
    def __init__(self, model, p_est_list=[]):
        Estimator.__init__(self, model)
        do_mpc.optimizer.Optimizer.__init__(self)

        # Parameters that can be set for the MHE:
        self.data_fields = [
            'n_horizon',
            't_step',
            'meas_from_data',
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
        self.meas_from_data = False
        self.state_discretization = 'collocation'
        self.collocation_type = 'radau'
        self.collocation_deg = 2
        self.collocation_ni = 1
        self.store_full_solution = False
        self.store_lagr_multiplier = True
        self.store_solver_stats = [
            'success',
            't_wall_S',
            't_wall_S',
        ]
        self.nlpsol_opts = {} # Will update default options with this dict.


        # Create seperate structs for the estimated and the set parameters (the union of both are all parameters of the model.)
        _p = model._p
        self._p_est  = struct_symSX(
            [entry('default', shape=(0,0))]+
            [entry(p_i, sym=_p[p_i]) for p_i in _p.keys() if p_i in p_est_list]
        )
        self._p_set  = struct_symSX(
            [entry(p_i, sym=_p[p_i]) for p_i in _p.keys() if p_i not in p_est_list]
        )
        # Function to obtain full set of parameters from the seperate structs (while obeying the order):
        self._p_cat_fun = Function('p_cat_fun', [self._p_est, self._p_set], [_p])

        self.n_p_est = self._p_est.shape[0]
        self.n_p_set = self._p_set.shape[0]

        # Initialize additional structures by calling the symbolic structures defined above
        # with the default numerical value.
        # This returns an identical numerical structure with all values set to the passed value.
        # TODO: p_scaling already exists. Maybe use it instead of these seperate structs?
        self._p_est_scaling = self._p_est(1.0)
        self._p_set_scaling = self._p_set(1.0) # This not meant to be adapted. We need it to concatenate p_scaling.

        self._p_est_lb = self._p_est(-np.inf)
        self._p_est_ub = self._p_est(np.inf)

        self._p_est0 = self._p_est(0.0)


        # Introduce aliases / new variables to smoothly and intuitively formulate
        # the MHE objective function.
        self._y_meas = self.model._y
        self._y_calc = self.model._y_expression

        self._x_prev = copy.copy(self.model._x)
        self._x = self.model._x

        self._p_est_prev = copy.copy(self._p_est)
        self._p_est = self._p_est

        # Flags are checked when calling .setup.
        self.flags = {
            'setup': False,
            'set_tvp_fun': False,
            'set_p_fun': False,
            'set_y_fun': False,
            'set_objective': False,
        }

    # @IndexedProperty
    # def vars(self, ind):
    #     if isinstance(ind, tuple):
    #         assert ind[0] in self.__dict__.keys(), '{} is not a MHE variable.'.format(ind[0])
    #         rval = self.__dict__[ind[0]][ind[1:]]
    #     elif isinstance(ind, str):
    #         assert ind in self.__dict__.keys(), '{} is not a MHE variable.'.format(ind)
    #         rval = self.__dict__[ind]
    #     else:
    #         raise Exception('Index {} is not valid.'.format(ind))
    #     return rval
    #
    # @vars.setter
    # def vars(self, ind, val):
    #     raise Exception('Setting MHE variables is not allowed.')


    def set_param(self, **kwargs):
        """Method to set the parameters of the :py:class:`MHE` class. Parameters must be passed as pairs of valid keywords and respective argument.
        For example:

        ::

            mhe.set_param(n_horizon = 20)

        It is also possible and convenient to pass a dictionary with multiple parameters simultaneously as shown in the following example:

        ::

            setup_mhe = {
                'n_horizon': 20,
                't_step': 0.5,
            }
            mhe.set_param(**setup_mhe)

        .. note:: :py:func:`mhe.set_param` can be called multiple times. Previously passed arguments are overwritten by successive calls.

        The following parameters are available:

        :param n_horizon: Prediction horizon of the optimal control problem. Parameter must be set by user.
        :type n_horizon: int

        :param t_step: Timestep of the mhe.
        :type t_step: float

        :param meas_from_data: Default option to retrieve past measurements for the MHE optimization problem. The :py:func:`MHE.set_y_fun` is called during setup.
        :type meas_from_data: bool

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

        .. note:: We highly suggest to change the linear solver for IPOPT from `mumps` to `MA27`. In many cases this will drastically boost the speed of **do-mpc**. Change the linear solver with:

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


    def set_objective(self, stage_cost, arrival_cost):
        """Set the objective function for the MHE problem. We suggest to formulate the MHE objective (:math:`J`) such that:

        .. math::

           J= & \\underbrace{(x_0 - \\tilde{x}_0)^T P_x (x_0 - \\tilde{x}_0)}_{\\text{arrival cost states}} +
           \\underbrace{(p_0 - \\tilde{p}_0)^T P_p (p_0 - \\tilde{p}_0)}_{\\text{arrival cost params.}} \\\\
           & +\sum_{k=0}^{n-1} \\underbrace{(h(x_k, u_k, p_k) - y_k)^T P_{y,k} (h(x_k, u_k, p_k) - y_k)}_{\\text{stage cost}}

        Use the class attributes:

        * ``mhe._y_meas`` as :math:`y_k`

        * ``mhe._y_calc`` as :math:`h(x_k, u_k, p_k)` (function is defined in :py:class:`do_mpc.model.Model`)

        * ``mhe._x_prev`` as :math:`\\tilde{x}_0`

        * ``mhe._x`` as :math:`x_0`

        * ``mhe._p_est_prev`` as :math:`\\tilde{p}_0`

        * ``mhe._p_est`` as :math:`p_0`

        To formulate the objective function and pass the stage cost and arrival cost independently.

        .. note::
            The retrieved attributes are symbolic structures, which can be queried with the given variable names,
            e.g.:

            ::

                x1 = mhe._x['state_1']

            For a vector of all states, use the ``.cat`` method as shown in the example below.

        **Example:**

        ::

            # Get variables:
            y_meas = mhe._y_meas
            y_calc = mhe._y_calc

            dy = y_meas.cat-y_calc.cat
            stage_cost = dy.T@np.diag(np.array([1,1,1,20,20]))@dy

            x_0 = mhe._x
            x_prev = mhe._x_prev
            p_0 = mhe._p_est
            p_prev = mhe._p_est_prev

            dx = x_0.cat - x_prev.cat
            dp = p_0.cat - p_prev.cat

            arrival_cost = 1e-4*dx.T@dx + 1e-4*dp.T@dp

            mhe.set_objective(stage_cost, arrival_cost)

        .. note::
            Use :py:func:`MHE.set_default_objective` as a high-level wrapper for this method,
            if you want to use the default MHE objective function.

        :param stage_cost: Stage cost that is added to the MHE objective at each age.
        :type stage_cost: CasADi expression

        :param arrival_cost: Arrival cost that is added to the MHE objective at the initial state.
        :type arrival_cost: CasADi expression

        :return: None
        :rtype: None
        """
        assert stage_cost.shape == (1,1), 'stage_cost must have shape=(1,1). You have {}'.format(stage_cost.shape)
        assert arrival_cost.shape == (1,1), 'arrival_cost must have shape=(1,1). You have {}'.format(arrival_cost.shape)
        assert self.flags['setup'] == False, 'Cannot call .set_objective after .setup.'


        stage_cost_input = self.model._x, self.model._u, self.model._z, self.model._tvp, self.model._p, self._y_meas
        assert set(symvar(stage_cost)).issubset(set(symvar(vertcat(*stage_cost_input)))), 'objective cost equation must be solely depending on x, u, z, p, tvp, y_meas.'
        self.stage_cost_fun = Function('stage_cost_fun', [*stage_cost_input], [stage_cost])

        arrival_cost_input = self._x, self._x_prev, self._p_est, self._p_est_prev
        assert set(symvar(arrival_cost)).issubset(set(symvar(vertcat(*arrival_cost_input)))), 'Arrival cost equation must be solely depending on x_0, x_prev, p_0, p_prev.'
        self.arrival_cost_fun = Function('arrival_cost_fun', arrival_cost_input, [arrival_cost])

        self.flags['set_objective'] = True

    def set_default_objective(self, P_x, P_y, P_p=None):
        """ Wrapper function to set the suggested default MHE objective:

        .. math::

           J= & \\underbrace{(x_0 - \\tilde{x}_0)^T P_x (x_0 - \\tilde{x}_0)}_{\\text{arrival cost states}} +
           \\underbrace{(p_0 - \\tilde{p}_0)^T P_p (p_0 - \\tilde{p}_0)}_{\\text{arrival cost params.}} \\\\
           & +\\sum_{k=0}^{n-1} \\underbrace{(h(x_k, u_k, p_k) - y_k)^T P_{y,k} (h(x_k, u_k, p_k) - y_k)}_{\\text{stage cost}}

        Pass the weighting matrices :math:`P_x`, :math:`P_p` and :math:`P_y`.
        The matrices must be of appropriate dimension (and numpy nd.arrays).
        In the case that no parameters are estimated, the weighting matrix :math:`P_p` is not required.

        .. note::
            Use :py:func:`MHE.set_objective` as a low-level alternative for this method,
            if you want to use a custom objective function.

        :param P_x: Tuning matrix :math:`P_x` of dimension :math:`n \\times n` (:math:`x \\in \\mathbb{R}^{n}`)
        :type P_x: numpy.ndarray
        :param P_y: Tuning matrix :math:`P_y` of dimension :math:`m \\times m` (:math:`y \\in \\mathbb{R}^{m}`)
        :type P_y: numpy.ndarray
        :param P_p: Tuning matrix :math:`P_p` of dimension :math:`l \\times l` (:math:`p_{\text{est}} \\in \\mathbb{R}^{l}`)
        :type P_p: numpy.ndarray, optional
        """

        assert isinstance(P_x, np.ndarray), 'P_x must be of type numpy.ndarray'
        assert isinstance(P_y, np.ndarray), 'P_y must be of type numpy.ndarray'
        assert isinstance(P_p, (np.ndarray, type(None))), 'P_p must be of type numpy.ndarray or None object.'
        n_x = self.model.n_x
        n_y = self.model.n_y
        n_p = self.n_p_est
        assert P_x.shape == (n_x, n_x), 'P_x has wrong shape:{}, must be {}'.format(P_x.shape, (n_x,n_x))
        assert P_y.shape == (n_y, n_y), 'P_y has wrong shape:{}, must be {}'.format(P_y.shape, (n_y,n_y))


        # Calculate stage cost:
        y_meas = self._y_meas
        y_calc = self._y_calc
        dy = y_meas.cat-y_calc.cat

        stage_cost = dy.T@P_y@dy

        # Calculate arrival cost:
        x_0 = self._x
        x_prev = self._x_prev
        dx = x_0.cat - x_prev.cat

        arrival_cost = dx.T@P_x@dx

        # Add parameter term if there are parameters to be estimated:
        if P_p is None:
            assert n_p == 0, 'Must pass weighting factor P_p, since you are trying to estimate parameters.'
        else:
            assert P_p.shape == (n_p, n_p), 'P_p has wrong shape:{}, must be {}'.format(P_p.shape, (n_p,n_p))
            p_0 = self._p_est
            p_prev = self._p_est_prev
            dp = p_0.cat - p_prev.cat
            arrival_cost += dp.T@P_p@dp

        # Set MHE objective:
        self.set_objective(stage_cost, arrival_cost)




    def get_p_template(self):
        """Obtain the a numerical copy of the structure of the (not estimated) parameters.
        Use this structure as the return of a user defined parameter function (``p_fun``)
        that is called at each MHE step. Pass this function to the MHE by calling :py:func:`MHE.set_p_fun`.

        .. note::
            The combination of :py:func:`MHE.get_p_template` and :py:func:`MHE.set_p_fun` is
            identical to the :py:class:`do_mpc.simulator.Simulator` methods, if the MHE
            is not estimating any parameters.

        :return: p_template
        :rtype: struct_symSX
        """
        return self._p_set(0)

    def set_p_fun(self, p_fun):
        """Set the parameter function which is called at each MHE time step and returns the (not) estimated parameters.
        The function must return a numerical CasADi structure, which can be retrieved with :py:func:`MHE.get_p_template`.

        :param p_fun: Parameter function.
        :type p_fun: function
        """
        assert self.get_p_template().labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.p_fun = p_fun
        self.flags['set_p_fun'] = True

    def get_y_template(self):
        """Obtain the a numerical copy of the structure of the measurements for the set horizon.
        Use this structure as the return of a user defined parameter function (``y_fun``)
        that is called at each MHE step. Pass this function to the MHE by calling :py:func:`MHE.set_y_fun`.

        The structure carries a set of measurements for each time step of the horizon and can be accessed as follows:

        ::

            y_template['y_meas', k, 'meas_name']
            # Slicing is possible, e.g.:
            y_template['y_meas', :, 'meas_name']

        where ``k`` runs from ``0`` to ``N_horizon-1`` and ``meas_name`` refers to the user-defined names in :py:class:`do_mpc.model.Model`.

        .. note::
            The structure is ordered, sucht that ``k=0`` is the "oldest measurement" and ``k=N_horizon-1`` is the newest measurement.

        By default, the following measurement function is choosen:

        ::

            y_template = self.get_y_template()

            def y_fun(t_now):
                n_steps = min(self.data._y.shape[0], self.n_horizon)
                for k in range(-n_steps,0):
                    y_template['y_meas',k] = self.data._y[k]
                try:
                    for k in range(self.n_horizon-n_steps):
                        y_template['y_meas',k] = self.data._y[-n_steps]
                except:
                    None
                return y_template

        Which simply reads the last results from the ``MHE.data`` object.

        :return: y_template
        :rtype: struct_symSX
        """
        y_template = struct_symSX([
            entry('y_meas', repeat=self.n_horizon, struct=self._y_meas)
        ])
        return y_template(0)

    def set_y_fun(self, y_fun):
        """Set the measurement function. The function must return a CasADi structure which can be obtained
        from :py:func:`MHE.get_y_template`. See the respective doc string for details.

        :param y_fun: measurement function.
        :type y_fun: function
        """
        assert self.get_y_template().labels() == y_fun(0).labels(), 'Incorrect output of y_fun. Use get_y_template to obtain the required structure.'
        self.y_fun = y_fun
        self.flags['set_y_fun'] = True


    def _check_validity(self):
        """Private method to be called in :py:func:`MHE.setup`. Checks if the configuration is valid and
        if the optimization problem can be constructed.
        Furthermore, default values are set if they were not configured by the user (if possible).
        Specifically, we set dummy values for the ``tvp_fun`` and ``p_fun`` if they are not present in the model
        and the default measurement function.
        """
        # Objective mus be defined.
        if self.flags['set_objective'] == False:
            raise Exception('Objective is undefined. Please call .set_objective() prior to .setup().')

        # tvp_fun must be set, if tvp are defined in model.
        if self.flags['set_tvp_fun'] == False and self.model._tvp.size > 0:
            raise Exception('You have not supplied a function to obtain the time varying parameters defined in model. Use .set_tvp_fun() prior to setup.')
        # p_fun must be set, if p are defined in model.
        if self.flags['set_p_fun'] == False and self._p_set.size > 0:
            raise Exception('You have not supplied a function to obtain the parameters defined in model. Use .set_p_fun() (low-level API) or .set_uncertainty_values() (high-level API) prior to setup.')


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

        if self.flags['set_y_fun'] == False and self.meas_from_data:
            # Case that measurement function is automatically created.
            y_template = self.get_y_template()

            def y_fun(t_now):
                n_steps = min(self.data._y.shape[0], self.n_horizon)
                for k in range(-n_steps,0):
                    y_template['y_meas',k] = self.data._y[k]
                try:
                    for k in range(self.n_horizon-n_steps):
                        y_template['y_meas',k] = self.data._y[-n_steps]
                except:
                    None
                return y_template
            self.set_y_fun(y_fun)
        elif self.flags['set_y_fun'] == True:
            # Case that the user supplied a measurement function.
            pass
        else:
            # No measurement function.
            raise Exception('You have not suppplied a measurement function. Use .set_y_fun or set parameter meas_from_data to True for default function.')


    def set_initial_guess(self):
        """Uses the current class attributes ``_x0``, ``_z0`` and ``_u0``, ``_p_est0`` to create an initial guess for the mhe.
        The initial guess is simply the initial values for all instances of x, u and z, p_est. The method is automatically
        evoked when calling the :py:func:`MHE.setup` method.
        However, if no initial values for x, u and z were supplied during setup, these default to zero.
        """
        assert self.flags['setup'] == True, 'mhe was not setup yet. Please call mhe.setup().'

        self.opt_x_num['_x'] = self._x0.cat/self._x_scaling
        self.opt_x_num['_u'] = self._u0.cat/self._u_scaling
        self.opt_x_num['_z'] = self._z0.cat/self._z_scaling
        self.opt_x_num['_p_est'] = self._p_est0.cat/self._p_est_scaling

    def setup(self):
        """The setup method finalizes the MHE creation. After this call, the :py:func:`do_mpc.optimizer.Optimizer.solve` method is applicable.
        The method wraps the following calls:

        * :py:func:`do_mpc.optimizer.Optimizer._setup_nl_cons`

        * :py:func:`MHE._check_validity`

        * :py:func:`MHE._setup_mhe_optim_problem`

        * :py:func:`MHE.set_initial_guess`


        and sets the setup flag = True.

        """
        self._setup_nl_cons()

        # Concatenate _p_est_scaling und _p_set_scaling to p_scaling (and make it a struct again)
        self._p_scaling = self.model._p(self._p_cat_fun(self._p_est_scaling, self._p_set_scaling))

        # Gather meta information:
        meta_data = {key: getattr(self, key) for key in self.data_fields}
        self.data.set_meta(**meta_data)

        self._check_validity()
        self._setup_mhe_optim_problem()
        self.flags['setup'] = True

        self.set_initial_guess()
        self._prepare_data()

    def make_step(self, y0):
        """Main method of the class during runtime. This method is called at each timestep
        and returns the current state estimate for the current measurement ``y0``.

        The method prepares the MHE by setting the current parameters, calls :py:func:`do_mpc.optimizer.Optimizer.solve`
        and updates the :py:class:`do_mpc.data.Data` object.

        :param y0: Current measurement.
        :type y0: numpy.ndarray

        :return: x0, estimated state of the system.
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'ME was not setup yet. Please call ME.setup().'

        self.data.update(_y = y0)


        p_est0 = self._p_est0
        x0 = self._x0

        t0 = self._t0
        tvp0 = self.tvp_fun(t0)
        p_set0 = self.p_fun(t0)

        y_traj = self.y_fun(t0)

        self.opt_p_num['_x_prev'] = self.opt_x_num['_x', 1, -1]*self._x_scaling
        self.opt_p_num['_p_est_prev'] = p_est0
        self.opt_p_num['_p_set'] = p_set0
        self.opt_p_num['_tvp'] = tvp0['_tvp']
        self.opt_p_num['_y_meas'] = y_traj['y_meas']

        self.solve()

        # Extract solution:
        x_next = self.opt_x_num['_x', -1, -1]*self._x_scaling
        p_est_next = self.opt_x_num['_p_est']*self._p_est_scaling
        u0 = self.opt_x_num['_u', -1]*self._u_scaling
        z0  = self.opt_x_num['_z', -1, -1]*self._z_scaling
        aux0 = self.opt_aux_num['_aux', -1]
        p0 = self._p_cat_fun(p_est0, p_set0)

        # Update data object:
        self.data.update(_x = x0)
        self.data.update(_u = u0)
        self.data.update(_z = z0)
        self.data.update(_p = p0)
        self.data.update(_tvp = tvp0['_tvp', -1])
        self.data.update(_time = t0)
        self.data.update(_aux = aux0)

        # Store additional information
        self.data.update(opt_p_num = self.opt_p_num)
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
        self._x0.master = x_next
        self._p_est0.master = p_est_next
        self._u0.master = u0
        self._z0.master = z0

        return x_next.full()

    def _setup_mhe_optim_problem(self):
        """Private method of the MHE class to construct the MHE optimization problem.
        The method depends on inherited methods from the :py:class:`do_mpc.optimizer.Optimizer`,
        such as :py:func:`do_mpc.optimizer.Optimizer._setup_discretization` and
        :py:func:`do_mpc.optimizer.Optimizer._setup_scenario_tree`.

        The MPC has a similar method with similar structure.
        """
        # Obtain an integrator (collocation, discrete-time) and the amount of intermediate (collocation) points
        ifcn, n_total_coll_points = self._setup_discretization()
        # Create struct for optimization variables:
        self.opt_x = opt_x = struct_symSX([
            entry('_x', repeat=[self.n_horizon+1, 1+n_total_coll_points], struct=self.model._x),
            entry('_z', repeat=[self.n_horizon,   1+n_total_coll_points], struct=self.model._z),
            entry('_u', repeat=[self.n_horizon], struct=self.model._u),
            entry('_eps', repeat=[self.n_horizon], struct=self._eps),
            entry('_p_est', struct=self._p_est),
        ])
        self.n_opt_x = self.opt_x.shape[0]
        # NOTE: The entry _x[k,:] starts with the collocation points from s to b at time k
        #       and the last point contains the child node
        # NOTE: Currently there exist dummy collocation points for the initial state (for each branch)

        # Create scaling struct as assign values for _x, _u, _z.
        self.opt_x_scaling = opt_x_scaling = opt_x(1)
        opt_x_scaling['_x'] = self._x_scaling
        opt_x_scaling['_z'] = self._z_scaling
        opt_x_scaling['_u'] = self._u_scaling
        opt_x_scaling['_p_est'] = self._p_est_scaling
        # opt_x are unphysical (scaled) variables. opt_x_unscaled are physical (unscaled) variables.
        self.opt_x_unscaled = opt_x_unscaled = opt_x(opt_x.cat * opt_x_scaling)


        # Create struct for optimization parameters:
        self.opt_p = opt_p = struct_symSX([
            entry('_x_prev', struct=self.model._x),
            entry('_p_est_prev', struct=self._p_est_prev),
            entry('_p_set', struct=self._p_set),
            entry('_tvp', repeat=self.n_horizon, struct=self.model._tvp),
            entry('_y_meas', repeat=self.n_horizon, struct=self.model._y),
        ])
        self.n_opt_p = opt_p.shape[0]

        # Dummy struct with symbolic variables
        self.aux_struct = struct_symSX([
            entry('_aux', repeat=[self.n_horizon], struct=self.model._aux_expression)
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

        # Arrival cost:
        arrival_cost = self.arrival_cost_fun(
            opt_x_unscaled['_x', 0, -1],
            opt_p['_x_prev'],#/self._x_scaling,
            opt_x_unscaled['_p_est'],
            opt_p['_p_est_prev'],#/self._p_est_scaling
            )

        obj += arrival_cost

        # Get concatenated parameters vector containing the estimated and fixed parameters (scaled)
        _p = self._p_cat_fun(self.opt_x['_p_est'], self.opt_p['_p_set']/self._p_set_scaling)

        # For all control intervals
        for k in range(self.n_horizon):
            # Compute constraints and predicted next state of the discretization scheme
            [g_ksb, xf_ksb] = ifcn(opt_x['_x', k, -1], vertcat(*opt_x['_x', k+1, :-1]),
                                   opt_x['_u', k], vertcat(*opt_x['_z', k, :]), opt_p['_tvp', k], _p)

            # Add the collocation equations
            cons.append(g_ksb)
            cons_lb.append(np.zeros(g_ksb.shape[0]))
            cons_ub.append(np.zeros(g_ksb.shape[0]))

            # Add continuity constraints
            cons.append(xf_ksb - opt_x['_x', k+1, -1])
            cons_lb.append(np.zeros((self.model.n_x, 1)))
            cons_ub.append(np.zeros((self.model.n_x, 1)))

            # Add nonlinear constraints only on each control step
            nl_cons_k = self._nl_cons_fun(
                opt_x_unscaled['_x', k, -1], opt_x_unscaled['_u', k], opt_x_unscaled['_z', k, -1],
                opt_p['_tvp', k], _p, opt_x_unscaled['_eps', k])

            cons.append(nl_cons_k)
            cons_lb.append(self._nl_cons_lb)
            cons_ub.append(self._nl_cons_ub)


            obj += self.stage_cost_fun(
                opt_x_unscaled['_x', k+1, -1], opt_x_unscaled['_u', k], opt_x_unscaled['_z', k, -1],
                opt_p['_tvp', k], _p, opt_p['_y_meas', k]
            )
            # Add slack variables to the cost
            obj += self.epsterm_fun(opt_x_unscaled['_eps', k])


            # Calculate the auxiliary expressions for the current scenario:
            opt_aux['_aux', k] = self.model._aux_expression_fun(
                opt_x_unscaled['_x', k, -1], opt_x_unscaled['_u', k], opt_x_unscaled['_z', k, -1], opt_p['_tvp', k], _p)

            # Bounds for the states on all discretize values along the horizon
            self.lb_opt_x['_x', k] = self._x_lb.cat/self._x_scaling
            self.ub_opt_x['_x', k] = self._x_ub.cat/self._x_scaling

            # Bounds for the inputs along the horizon
            self.lb_opt_x['_u', k] = self._u_lb.cat/self._u_scaling
            self.ub_opt_x['_u', k] = self._u_ub.cat/self._u_scaling

            # Bounds for the slack variables along the horizon:
            self.lb_opt_x['_eps', k] = self._eps_lb.cat
            self.ub_opt_x['_eps', k] = self._eps_ub.cat

        # Bounds for the inputs along the horizon
        self.lb_opt_x['_p_est'] = self._p_est_lb.cat/self._p_est_scaling
        self.ub_opt_x['_p_est'] = self._p_est_ub.cat/self._p_est_scaling

        # Bounds for the states at final time:
        self.lb_opt_x['_x', self.n_horizon] = self._x_lb.cat/self._x_scaling
        self.ub_opt_x['_x', self.n_horizon] = self._x_ub.cat/self._x_scaling


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
