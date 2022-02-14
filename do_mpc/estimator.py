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
import warnings
import time

import do_mpc.optimizer
import do_mpc.data


class Estimator(do_mpc.model.IteratedVariables):
    """The Estimator base class. Used for :py:class:`StateFeedback`, :py:class:`EKF` and :py:class:`MHE`.
    This class cannot be used independently.

    .. note::
       The methods :py:func:`Estimator.set_initial_state` and :py:func:`Estimator.reset_history`
       are overwritten when using the :py:class:`MHE` by the methods defined in :py:class:`do_mpc.optimizer.Optimizer`.

    """
    def __init__(self, model):
        self.model = model
        do_mpc.model.IteratedVariables.__init__(self)

        assert model.flags['setup'] == True, 'Model for estimator was not setup. After the complete model creation call model.setup().'

        self.data = do_mpc.data.Data(model)
        self.data.dtype = 'Estimator'


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
    """Moving horizon estimator.

    For general information on moving horizon estimation, please read our `background article`_.

    .. _`background article`: ../theory_mhe.html

    The MHE estimator extends the :py:class:`do_mpc.optimizer.Optimizer` base class
    (which is also used for :py:class:`do_mpc.controller.MPC`), as well as the :py:class:`Estimator` base class.
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

    1. Use :py:func:`set_param` to configure the :py:class:`MHE`. See docstring for details.

    2. Set the objective of the control problem with :py:func:`set_default_objective` or use the low-level interface :py:func:`set_objective`.

    5. Set upper and lower bounds.

    6. Optionally, set further (non-linear) constraints with :py:func:`set_nl_cons`.

    7. Use :py:func:`get_p_template` and :py:func:`set_p_fun` to set the function for the (not estimated) parameters.

    8. Use :py:meth:`get_tvp_template` and :py:meth:`set_tvp_fun` to create a method to obtain new time-varying parameters at each iteration.

    9. To finalize the class configuration there are two routes. The default approach is to call :py:meth:`setup`. For deep customization use the combination of :py:meth:`prepare_nlp` and :py:meth:`create_nlp`. See graph below for an illustration of the process.


    .. graphviz::
        :name: route_to_setup
        :caption: Route to setting up the MHE class.
        :align: center

        digraph G {
            graph [fontname = "helvetica"];
            rankdir=LR;

            subgraph cluster_main {
                node [fontname = "helvetica", shape=box, fontcolor="#404040", color="#707070"];
                edge [fontname = "helvetica", color="#707070"];

                start [label="Two ways to setup"];
                setup [label="setup", href="../api/do_mpc.estimator.MHE.setup.html", target="_top", fontname = "Consolas"];
                create_nlp [label="create_nlp", href="../api/do_mpc.estimator.MHE.create_nlp.html", target="_top", fontname = "Consolas"];
                process [label="Modify NLP"];
                prepare_nlp [label="prepare_nlp", href="../api/do_mpc.estimator.MHE.prepare_nlp.html", target="_top", fontname = "Consolas"];
                finish [label="Configured MHE class"]
                start -> setup, prepare_nlp;
                prepare_nlp -> process;
                process -> create_nlp;
                setup, create_nlp -> finish;
                color=none;
            }

            subgraph cluster_modification {
                rankdir=TB;
                node [fontname = "helvetica", shape=box, fontcolor="#404040", color="#707070"];
                edge [fontname = "helvetica", color="#707070"];
                opt_x [label="opt_x", href="../api/do_mpc.estimator.MHE.opt_x.html", target="_top", fontname = "Consolas"];
                opt_p [label="opt_p", href="../api/do_mpc.estimator.MHE.opt_p.html", target="_top", fontname = "Consolas"];
                nlp_cons [label="nlp_cons", href="../api/do_mpc.estimator.MHE.nlp_cons.html", target="_top", fontname = "Consolas"];
                nlp_obj [label="nlp_obj", href="../api/do_mpc.estimator.MHE.nlp_obj.html", target="_top", fontname = "Consolas"];

                opt_x -> nlp_cons, nlp_obj;
                opt_p -> nlp_cons, nlp_obj;

                label = "Attributes to modify the NLP.";
		        color=black;
            }

            nlp_cons -> process;
            nlp_obj -> process;
        }

    .. warning::

        Before running the estimator, make sure to supply a valid initial guess for all estimated variables (states, algebraic states, inputs and parameters).
        Simply set the intial values of :py:attr:`x0`, :py:attr:`z0`, :py:attr:`u0` and :py:attr:`p_est0` and then call :py:func:`set_initial_guess`.

        To take full control over the initial guess, modify the values of :py:attr:`opt_x_num`.

    During runtime use :py:func:`make_step` with the most recent measurement to obtain the estimated states.

    :param model: A configured and setup :py:class:`do_mpc.model.Model`
    :type model: :py:class:`do_mpc.model.Model`

    :param p_est_list: List with names of parameters (``_p``) defined in ``model``
    :type p_est_list: list

    """

    def __init__(self, model, p_est_list=[]):
        Estimator.__init__(self, model)
        do_mpc.optimizer.Optimizer.__init__(self)

        # Initialize structure to hold the optimial solution and initial guess:
        self._opt_x_num = None
        # Initialize structure to hold the parameters for the optimization problem:
        self._opt_p_num = None

        # Parameters that can be set for the MHE:
        self.data_fields = [
            'n_horizon',
            't_step',
            'meas_from_data',
            'state_discretization',
            'collocation_type',
            'collocation_deg',
            'collocation_ni',
            'nl_cons_check_colloc_points',
            'nl_cons_single_slack',
            'cons_check_colloc_points',
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
        self.nl_cons_check_colloc_points = False
        self.nl_cons_single_slack = False
        self.cons_check_colloc_points = True
        self.store_full_solution = False
        self.store_lagr_multiplier = True
        self.store_solver_stats = [
            'success',
            't_wall_total',
        ]
        self.nlpsol_opts = {} # Will update default options with this dict.


        # Create seperate structs for the estimated and the set parameters (the union of both are all parameters of the model.)
        _p = model._p
        self._p_est  = self.model.sv.sym_struct(
            [entry('default', shape=(0,0))]+
            [entry(p_i, shape=_p[p_i].shape) for p_i in _p.keys() if p_i in p_est_list]
        )
        self._p_set  = self.model.sv.sym_struct(
            [entry(p_i, shape=_p[p_i].shape) for p_i in _p.keys() if p_i not in p_est_list]
        )

        # Enable to "unite" _p_est and _p_set to _p
        p_cat = vertcat(_p)
        _p_subs = []
        for name in _p.keys():
            if name in self._p_est.keys():
                _p_subs.append(self._p_est[name])
            elif name in self._p_set.keys():
                _p_subs.append(self._p_set[name])

        # In the expression p_cat substitute all variables from _p with the elements from _p_est and _p_set:
        p_cat = substitute(p_cat, _p, vertcat(*_p_subs))

        # Function to obtain full set of parameters from the seperate structs (while obeying the order):
        self._p_cat_fun = Function('p_cat_fun', [self._p_est, self._p_set], [p_cat])


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

        self._x_prev = copy.copy(self.model._x)
        self._x = self.model._x

        self._p_est_prev = copy.copy(self._p_est)
        self._p = self.model._p

        self._w = self.model._w
        self._v = self.model._v

        # Flags are checked when calling .setup.
        self.flags.update({
            'setup': False,
            'set_tvp_fun': False,
            'set_p_fun': False,
            'set_y_fun': False,
            'set_objective': False,
            'set_initial_guess': False,
        })

    @property
    def p_est0(self):
        """ Initial value of estimated parameters and current iterate.
        This is the numerical structure holding the information about the current
        estimated parameters in the class.
        The property can be indexed according to the model definition.

        **Example:**

        ::

            model = do_mpc.model.Model('continuous')
            model.set_variable('_p','temperature', shape=(4,1))

            # Initiate MHE with list of estimated parameters:
            mhe = do_mpc.estimator.MHE(model, ['temperature'])

            # Get or set current value of variable:
            mhe.p_est0['temperature', 0] # 0th element of variable
            mhe.p_est0['temperature']    # all elements of variable
            mhe.p_est0['temperature', 0:2]    # 0th and 1st element

        Usefull CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``

        """
        return self._p_est0

    @p_est0.setter
    def p_est0(self, val):
        self._p_est0 = self._convert2struct(val, self._p_est)


    @property
    def opt_x_num(self):
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

            The attribute is populated when calling :py:func:`setup`
        """
        return self._opt_x_num

    @opt_x_num.setter
    def opt_x_num(self, val):
        self._opt_x_num = val

    @property
    def opt_p_num(self):
        """Full MHE parameter vector.

        This attribute is used when calling the solver to pass all required parameters,
        including

        * previously estimated state(s)

        * previously estimated parameter(s)

        * known parameters

        * sequence of time-varying parameters

        * sequence of measurements parameters

        **do-mpc** handles setting these parameters automatically in the :py:func:`make_step`
        method. However, you can set these values manually and directly call :py:func:`solve`.

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

            The attribute is populated when calling :py:func:`setup`

        """
        return self._opt_p_num

    @opt_p_num.setter
    def opt_p_num(self, val):
        self._opt_p_num = val

    @property
    def opt_x(self):
        """Full structure of the (symbolic) MHE optimization variables.

        The attribute is a CasADi numeric structure with nested power indices.
        It can be indexed as follows:

        ::

            # dynamic states:
            opt_x['_x', time_step, collocation_point, _x_name]
            # algebraic states:
            opt_x['_z', time_step, collocation_point, _z_name]
            # inputs:
            opt_x['_u', time_step, _u_name]
            # estimated parameters:
            opt_x_Num['_p_est', _p_names]
            # slack variables for soft constraints:
            opt_x['_eps', time_step, _nl_cons_name]

        The names refer to those given in the :py:class:`do_mpc.model.Model` configuration.
        Further indices are possible, if the variables are itself vectors or matrices.

        The attribute can be used to alter the objective function or constraints of the NLP.

        .. note::

            The attribute ``opt_x`` carries the scaled values of all variables.

        .. warning::

            Do not tweak or overwrite this attribute unless you known what you are doing.

        .. note::

            The attribute is populated when calling :py:func:`setup` or :py:func:`prepare_nlp`

        """
        return self._opt_x

    @opt_x.setter
    def opt_x(self, val):
        self._opt_x = val

    @property
    def opt_p(self):
        """Full structure of (symbolic) MHE parameters.

        The attribute can be used to alter the objective function or constraints of the NLP.

        The attribute is a CasADi numeric structure with nested power indices.
        It can be indexed as follows:

        ::

            # previously estimated state:
            opt_p['_x_prev', _x_name]
            # previously estimated parameters:
            opt_p['_p_est_prev', _x_name]
            # known parameters
            opt_p['_p_set', _p_name]
            # time-varying parameters:
            opt_p['_tvp', time_step, _tvp_name]
            # sequence of measurements:
            opt_p['_y_meas', time_step, _y_name]

        The names refer to those given in the :py:class:`do_mpc.model.Model` configuration.
        Further indices are possible, if the variables are itself vectors or matrices.

        .. warning::

            Do not tweak or overwrite this attribute unless you known what you are doing.

        .. note::

            The attribute is populated when calling :py:func:`setup` or :py:func:`create_nlp`.

        """
        return self._opt_p

    @opt_p.setter
    def opt_p(self, val):
        self._opt_p = val

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

        This makes use of thy python "unpack" operator. See `more details here`_.

        .. _`more details here`: https://codeyarns.github.io/tech/2012-04-25-unpack-operator-in-python.html

        .. note:: The only required parameters  are ``n_horizon`` and ``t_step``. All other parameters are optional.

        .. note:: :py:func:`set_param` can be called multiple times. Previously passed arguments are overwritten by successive calls.

        The following parameters are available:

        :param n_horizon: Prediction horizon of the optimal control problem. Parameter must be set by user.
        :type n_horizon: int

        :param t_step: Timestep of the mhe.
        :type t_step: float

        :param meas_from_data: Default option to retrieve past measurements for the MHE optimization problem. The :py:func:`set_y_fun` is called during setup.
        :type meas_from_data: bool

        :param state_discretization: Choose the state discretization for continuous models. Currently only ``'collocation'`` is available. Defaults to ``'collocation'``. Has no effect if model is created in ``discrete`` type.
        :type state_discretization: str

        :param collocation_type: Choose the collocation type for continuous models with collocation as state discretization. Currently only ``'radau'`` is available. Defaults to ``'radau'``.
        :type collocation_type: str

        :param collocation_deg: Choose the collocation degree for continuous models with collocation as state discretization. Defaults to ``2``.
        :type collocation_deg: int

        :param collocation_ni: For orthogonal collocation, choose the number of finite elements for the states within a time-step (and during constant control input). Defaults to ``1``. Can be used to avoid high-order polynomials.
        :type collocation_ni: int

        :param nl_cons_check_colloc_points: For orthogonal collocation choose wether the bounds set with :py:func:`set_nl_cons` are evaluated once per finite Element or for each collocation point. Defaults to ``False`` (once per collocation point).
        :type nl_cons_check_colloc_points: bool

        :param cons_check_colloc_points: For orthogonal collocation choose whether the linear bounds set with :py:attr:`bounds` are evaluated once per finite Element or for each collocation point. Defaults to ``True`` (for all collocation points).
        :type cons_check_colloc_points: bool

        :param nl_cons_single_slack: If ``True``, soft-constraints set with :py:func:`set_nl_cons` introduce only a single slack variable for the entire horizon. Defaults to ``False``.
        :type nl_cons_single_slack: bool

        :param store_full_solution: Choose whether to store the full solution of the optimization problem. This is required for animating the predictions in post processing. However, it drastically increases the required storage. Defaults to False.
        :type store_full_solution: bool

        :param store_lagr_multiplier: Choose whether to store the lagrange multipliers of the optimization problem. Increases the required storage. Defaults to ``True``.
        :type store_lagr_multiplier: bool

        :param store_solver_stats: Choose which solver statistics to store. Must be a list of valid statistics. Defaults to ``['success','t_wall_S']``.
        :type store_solver_stats: list

        :param nlpsol_opts: Dictionary with options for the CasADi solver call ``nlpsol`` with plugin ``ipopt``. All options are listed `here <http://casadi.sourceforge.net/api/internal/d4/d89/group__nlpsol.html>`_.
        :type store_solver_stats: dict

        .. note:: We highly suggest to change the linear solver for IPOPT from `mumps` to `MA27`. In many cases this will drastically boost the speed of **do-mpc**. Change the linear solver with:

            ::

                optimizer.set_param(nlpsol_opts = {'ipopt.linear_solver': 'MA27'})
        .. note:: To suppress the output of IPOPT, please use:

            ::

                suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
                optimizer.set_param(nlpsol_opts = suppress_ipopt)

        """
        assert self.flags['setup'] == False, 'Setting parameters after setup is prohibited.'

        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for optimizer.'.format(key))
            else:
                setattr(self, key, value)


    def set_objective(self, stage_cost, arrival_cost):
        """Set the stage cost :math:`l(\cdot)` and arrival cost :math:`m(\cdot)` function for the MHE problem:

        .. math::

            \\underset{
            \\begin{array}{c}
            \\mathbf{x}_{0:N+1}, \\mathbf{u}_{0:N}, p,\\\\
            \\mathbf{w}_{0:N}, \\mathbf{v}_{0:N}
            \\end{array}
            }{\\mathrm{min}}
            &m(x_0,\\tilde{x}_0, p,\\tilde{p})
            +\\sum_{k=0}^{N-1} l(v_k, w_k, p, p_{\\text{tv},k}),\\\\
            &\\left.\\begin{aligned}
            \\mathrm{s.t.}\\quad
            x_{k+1} &= f(x_k,u_k,z_k,p,p_{\\text{tv},k})+ w_k,\\\\
            y_k &= h(x_k,u_k,z_k,p,p_{\\text{tv},k}) + v_k, \\\\
            &g(x_k,u_k,z_k,p_k,p_{\\text{tv},k}) \\leq 0
            \\end{aligned}\\right\} k=0,\\dots, N

        Use the class attributes:

        * ``mhe._w`` as :math:`w_k`

        * ``mhe._v`` as :math:`v_k`

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
            v = mhe._v.cat

            stage_cost = v.T@np.diag(np.array([1,1,1,20,20]))@v

            x_0 = mhe._x
            x_prev = mhe._x_prev
            p_0 = mhe._p_est
            p_prev = mhe._p_est_prev

            dx = x_0.cat - x_prev.cat
            dp = p_0.cat - p_prev.cat

            arrival_cost = 1e-4*dx.T@dx + 1e-4*dp.T@dp

            mhe.set_objective(stage_cost, arrival_cost)

        .. note::

            Use :py:func:`set_default_objective` as a high-level wrapper for this method,
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

        # Replace model symbolic variables self.model._p with the new variables self._p_est and self._p_set:
        _p = self._p_cat_fun(self._p_est, self._p_set)

        arrival_cost = substitute(arrival_cost, self.model._p, _p.reshape((-1,1)))
        #stage_cost = substitute(stage_cost, self.model._p, _p)

        #arrival_cost = substitute(arrival_cost, self._p_est, vertcat(*[self.model._p[name] for name in self._p_est.keys()]))
        #arrival_cost = substitute(arrival_cost, self._p_set, vertcat(*[self.model._p[name] for name in self._p_set.keys()]))

        stage_cost = substitute(stage_cost, self._p_est, vertcat(*[self.model._p[name] for name in self._p_est.keys()]).reshape((-1,1)))
        stage_cost = substitute(stage_cost, self._p_set, vertcat(*[self.model._p[name] for name in self._p_set.keys()]).reshape((-1,1)))


        stage_cost_input = self._w, self._v, self.model._tvp, self.model._p
        self.stage_cost_fun = Function('stage_cost_fun', [*stage_cost_input], [stage_cost])

        arrival_cost_input = self._x, self._x_prev, self._p_est, self._p_est_prev, self._p_set
        self.arrival_cost_fun = Function('arrival_cost_fun', [*arrival_cost_input], [arrival_cost])

        # Check if stage_cost_fun and arrival_cost_fun use invalid variables as inputs.
        # For the check we evaluate the function with dummy inputs and expect a DM output.
        try:
            self.stage_cost_fun(*[input_i(0) for input_i in stage_cost_input])
        except:
            err_msg = 'objective cost equation must be solely depending on w, v, p and tvp.'
            raise Exception(err_msg)
        try:
            self.arrival_cost_fun(*[input_i(0) for input_i in arrival_cost_input])
        except:
            err_msg = 'Arrival cost equation must be solely depending on x_0, x_prev, p_0, p_prev, p_set'
            raise Exception(err_msg)

        self.flags['set_objective'] = True

    def set_default_objective(self, P_x, P_v=None, P_p=None, P_w=None):
        """ Configure the suggested default MHE formulation.

        Use this method to pass tuning matrices for the MHE optimization problem:

        .. math::

            \\underset{
            \\begin{array}{c}
            \\mathbf{x}_{0:N+1}, \\mathbf{u}_{0:N}, p,\\\\
            \\mathbf{w}_{0:N}, \\mathbf{v}_{0:N}
            \\end{array}
            }{\\mathrm{min}}
            &m(x_0,\\tilde{x}_0, p,\\tilde{p})
            +\\sum_{k=0}^{N-1} l(v_k, w_k, p, p_{\\text{tv},k}),\\\\
            &\\left.\\begin{aligned}
            \\mathrm{s.t.}\\quad
            x_{k+1} &= f(x_k,u_k,z_k,p,p_{\\text{tv},k})+ w_k,\\\\
            y_k &= h(x_k,u_k,z_k,p,p_{\\text{tv},k}) + v_k, \\\\
            &g(x_k,u_k,z_k,p_k,p_{\\text{tv},k}) \\leq 0
            \\end{aligned}\\right\} k=0,\\dots, N

        where we introduce the bold letter notation,
        e.g. :math:`\mathbf{x}_{0:N+1}=[x_0, x_1, \dots, x_{N+1}]^T` to represent sequences and where
        :math:`\|x\|_P^2=x^T P x` denotes the :math:`P` weighted squared norm.

        Pass the weighting matrices :math:`P_x`, :math:`P_p` and :math:`P_v` and :math:`P_w`.
        The matrices must be of appropriate dimension and array-like.

        .. note::

            It is possible to pass parameters or time-varying parameters defined in the
            :py:class:`do_mpc.model.Model` as weighting.
            You'll probably choose time-varying parameters (``_tvp``) for ``P_v`` and ``P_w``
            and parameters (``_p``) for ``P_x`` and ``P_p``.
            Use :py:func:`set_p_fun` and :py:func:`set_tvp_fun` to configure how these values
            are determined at each time step.

        **General remarks:**

        * In the case that no parameters are estimated, the weighting matrix :math:`P_p` is not required.

        * In the case that the :py:class:`do_mpc.model.Model` is configured without process-noise (see :py:func:`do_mpc.model.Model.set_rhs`) the parameter ``P_w`` is not required.

        * In the case that the :py:class:`do_mpc.model.Model` is configured without measurement-noise (see :py:func:`do_mpc.model.Model.set_meas`) the parameter ``P_v`` is not required.

        The respective terms are not present in the MHE formulation in that case.

        .. note::

            Use :py:func:`set_objective` as a low-level alternative for this method,
            if you want to use a custom objective function.

        :param P_x: Tuning matrix :math:`P_x` of dimension :math:`n \\times n` :math:`(x \\in \\mathbb{R}^{n})`
        :type P_x: numpy.ndarray, casadi.SX, casadi.DM
        :param P_v: Tuning matrix :math:`P_v` of dimension :math:`m \\times m` :math:`(v \\in \\mathbb{R}^{m})`
        :type P_v: numpy.ndarray, casadi.SX, casadi.DM
        :param P_p: Tuning matrix :math:`P_p` of dimension :math:`l \\times l` :math:`(p_{\\text{est}} \\in \\mathbb{R}^{l})`)
        :type P_p: numpy.ndarray, casadi.SX, casadi.DM
        :param P_w: Tuning matrix :math:`P_w` of dimension :math:`k \\times k` :math:`(w \\in \\mathbb{R}^{k})`
        :type P_w: numpy.ndarray, casadi.SX, casadi.DM
        """

        input_types = (np.ndarray, casadi.SX, casadi.MX, casadi.DM)
        err_msg = '{name} must be of type {type_set}, you have {type_is}'
        assert isinstance(P_x, input_types), err_msg.format(name='P_x', type_set = input_types, type_is = type(P_x))
        input_types = (np.ndarray, casadi.SX, casadi.MX, casadi.DM, type(None))
        assert isinstance(P_v, input_types), err_msg.format(name='P_v', type_set = input_types, type_is = type(P_v))
        assert isinstance(P_p, input_types), err_msg.format(name='P_p', type_set = input_types, type_is = type(P_p))
        assert isinstance(P_w, input_types), err_msg.format(name='P_w', type_set = input_types, type_is = type(P_w))

        n_x = self.model.n_x
        n_y = self.model.n_y
        n_w = self.model.n_w
        n_v = self.model.n_v
        n_p = self.n_p_est
        assert P_x.shape == (n_x, n_x), 'P_x has wrong shape:{}, must be {}'.format(P_x.shape, (n_x,n_x))



        # Calculate stage cost:
        stage_cost = DM(0)

        if P_v is None:
            assert n_v == 0, 'Must pass weighting factor P_v, since you have measurement noise on some measurements (configured in model).'
        else:
            assert P_v.shape == (n_v, n_v), 'P_v has wrong shape:{}, must be {}'.format(P_v.shape, (n_v,n_v))
            v = self._v.cat
            stage_cost += v.T@P_v@v


        if P_w is None:
            assert n_w == 0, 'Must pass weighting factor P_w, since you have process noise on some states (configured in model).'
        else:
            assert P_w.shape == (n_w, n_w), 'P_w has wrong shape:{}, must be {}'.format(P_w.shape, (n_w,n_w))
            w = self._w.cat
            stage_cost += w.T@P_w@w


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
        """Obtain output template for :py:func:`set_p_fun`.
        This is used to set the (not estimated) parameters.
        Use this structure as the return of a user defined parameter function (``p_fun``)
        that is called at each MHE step. Pass this function to the MHE by calling :py:func:`set_p_fun`.

        .. note::
            The combination of :py:func:`get_p_template` and :py:func:`set_p_fun` is
            identical to the :py:class:`do_mpc.simulator.Simulator` methods, if the MHE
            is not estimating any parameters.

        :return: p_template
        :rtype: struct_symSX
        """
        return self._p_set(0)

    def set_p_fun(self, p_fun):
        """Set function which returns parameters..
        The ``p_fun`` is called at each MHE time step and returns the (fixed) parameters.
        The function must return a numerical CasADi structure, which can be retrieved with :py:func:`get_p_template`.

        :param p_fun: Parameter function.
        :type p_fun: function
        """
        assert self.get_p_template().labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.p_fun = p_fun
        self.flags['set_p_fun'] = True

    def get_y_template(self):
        """Obtain output template for :py:func:`set_y_fun`.

        Use this structure as the return of a user defined parameter function (``y_fun``)
        that is called at each MHE step. Pass this function to the MHE by calling :py:func:`set_y_fun`.

        The structure carries a set of measurements **for each time step of the horizon** and can be accessed as follows:

        ::

            y_template['y_meas', k, 'meas_name']
            # Slicing is possible, e.g.:
            y_template['y_meas', :, 'meas_name']

        where ``k`` runs from ``0`` to ``N_horizon`` and ``meas_name`` refers to the user-defined names in :py:class:`do_mpc.model.Model`.

        .. note::
            The structure is ordered, sucht that ``k=0`` is the "oldest measurement" and ``k=N_horizon`` is the newest measurement.

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
        y_template = self.model.sv.sym_struct([
            entry('y_meas', repeat=self.n_horizon, struct=self._y_meas)
        ])
        return y_template(0)

    def set_y_fun(self, y_fun):
        """Set the measurement function. The function must return a CasADi structure which can be obtained
        from :py:func:`get_y_template`. See the respective doc string for details.

        :param y_fun: measurement function.
        :type y_fun: function
        """
        assert self.get_y_template().labels() == y_fun(0).labels(), 'Incorrect output of y_fun. Use get_y_template to obtain the required structure.'
        self.y_fun = y_fun
        self.flags['set_y_fun'] = True


    def _check_validity(self):
        """Private method to be called in :py:func:`setup`. Checks if the configuration is valid and
        if the optimization problem can be constructed.
        Furthermore, default values are set if they were not configured by the user (if possible).
        Specifically, we set dummy values for the ``tvp_fun`` and ``p_fun`` if they are not present in the model
        and the default measurement function.
        """
        # Objective mus be defined.
        if self.flags['set_objective'] == False:
            raise Exception('Objective is undefined. Please call .set_objective() or .set_default_objective() prior to .setup().')

        # tvp_fun must be set, if tvp are defined in model.
        if self.flags['set_tvp_fun'] == False and self.model._tvp.size > 0:
            raise Exception('You have not supplied a function to obtain the time-varying parameters defined in model. Use .set_tvp_fun() prior to setup.')
        # p_fun must be set, if p are defined in model.
        if self.flags['set_p_fun'] == False and self._p_set.size > 0:
            raise Exception('You have not supplied a function to obtain the parameters defined in model. Use .set_p_fun() prior to setup.')


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
        """Initial guess for optimization variables.
        Uses the current class attributes :py:obj:`x0`, :py:obj:`z0` and :py:obj:`u0`, :py:obj:`p_est0` to create an initial guess for the MHE.
        The initial guess is simply the initial values for all :math:`k=0,\dots,N` instances of :math:`x_k`, :math:`u_k` and :math:`z_k`, :math:`p_{\\text{est,k}}`.

        .. warning::
            If no initial values for :py:attr:`x0`, :py:attr:`z0` and :py:attr:`u0` were supplied during setup, these default to zero.

        .. note::
            The initial guess is fully customizable by directly setting values on the class attribute:
            :py:attr:`opt_x_num`.
        """
        assert self.flags['setup'] == True, 'mhe was not setup yet. Please call mhe.setup().'

        self.opt_x_num['_x'] = self._x0.cat/self._x_scaling
        self.opt_x_num['_u'] = self._u0.cat/self._u_scaling
        self.opt_x_num['_z'] = self._z0.cat/self._z_scaling
        self.opt_x_num['_p_est'] = self._p_est0.cat/self._p_est_scaling

        self.flags['set_initial_guess'] = True

    def setup(self):
        """The setup method finalizes the MHE creation.
        The optimization problem is created based on the configuration of the module.

        .. note::

            After this call, the :py:func:`solve` and :py:func:`make_step` method is applicable.
        """

        self.prepare_nlp()
        self.create_nlp()


    def make_step(self, y0):
        """Main method of the class during runtime. This method is called at each timestep
        and returns the current state estimate for the current measurement ``y0``.

        The method prepares the MHE by setting the current parameters, calls :py:func:`solve`
        and updates the :py:class:`do_mpc.data.Data` object.

        .. warning::

            Moving horizon estimation will only work reliably once **a full sequence of measurements**
            corresponding to the set horizon ist available.

        :param y0: Current measurement.
        :type y0: numpy.ndarray

        :return: x0, estimated state of the system.
        :rtype: numpy.ndarray
        """
        # Check setup.
        assert self.flags['setup'] == True, 'optimizer was not setup yet. Please call optimizer.setup().'

        # Check input type.
        if isinstance(y0, (np.ndarray, casadi.DM)):
            pass
        elif isinstance(y0, structure3.DMStruct):
            y0 = y0.cat
        else:
            raise Exception('Invalid type {} for y0. Must be {}'.format(type(y0), (np.ndarray, casadi.DM, structure3.DMStruct)))

        # Check input shape.
        n_val = np.prod(y0.shape)
        assert n_val == self.model.n_y, 'Wrong input with shape {}. Expected vector with {} elements'.format(n_val, self.model.n_y)
        # Check (once) if the initial guess was supplied.
        if not self.flags['set_initial_guess']:
            warnings.warn('Intial guess for the optimizer was not set. The solver call is likely to fail.')
            time.sleep(5)
            # Since do-mpc is warmstarting, the initial guess will exist after the first call.
            self.flags['set_initial_guess'] = True

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
        # Which z must be extracted here?
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

    def _prepare_nlp(self):
        """Internal method. See detailed documentation in optimizer.prepare_nlp
        """

        nl_cons_input = self.model['x', 'u', 'z', 'tvp']
        nl_cons_input += [self._p_est, self._p_set]
        self._setup_nl_cons(nl_cons_input)
        self._check_validity()

        # Concatenate _p_est_scaling und _p_set_scaling to p_scaling (and make it a struct again)
        self._p_scaling = self.model._p(self._p_cat_fun(self._p_est_scaling, self._p_set_scaling))


        # Obtain an integrator (collocation, discrete-time) and the amount of intermediate (collocation) points
        ifcn, n_total_coll_points = self._setup_discretization()

        # How many slack variables (for soft constraints) are introduced over the horizon.
        if self.nl_cons_single_slack:
            n_eps = 1
        else:
            n_eps = self.n_horizon


        # Create struct for optimization variables:
        self._opt_x = opt_x = self.model.sv.sym_struct([
            entry('_x', repeat=[self.n_horizon+1, 1+n_total_coll_points], struct=self.model._x),
            entry('_z', repeat=[self.n_horizon,   max(n_total_coll_points,1)], struct=self.model._z),
            entry('_u', repeat=[self.n_horizon], struct=self.model._u),
            entry('_w', repeat=[self.n_horizon], struct=self.model._w),
            entry('_v', repeat=[self.n_horizon], struct=self.model._v),
            entry('_eps', repeat=[n_eps], struct=self._eps),
            entry('_p_est', struct=self._p_est),
        ])

        self.n_opt_x = opt_x.shape[0]
        # NOTE: The entry _x[k,:] starts with the collocation points from s to b at time k
        #       and the last point contains the child node
        # NOTE: Currently there exist dummy collocation points for the initial state (for each branch)

        # Create scaling struct as assign values for _x, _u, _z.
        self.opt_x_scaling = opt_x_scaling = opt_x(1)
        opt_x_scaling['_x'] = self._x_scaling
        # TODO: Should _w be scaled?
        opt_x_scaling['_z'] = self._z_scaling
        opt_x_scaling['_u'] = self._u_scaling
        opt_x_scaling['_p_est'] = self._p_est_scaling
        # opt_x are unphysical (scaled) variables. opt_x_unscaled are physical (unscaled) variables.
        self.opt_x_unscaled = opt_x_unscaled = opt_x(opt_x.cat * opt_x_scaling)


        # Create struct for optimization parameters:
        self._opt_p = opt_p = self.model.sv.sym_struct([
            entry('_x_prev', struct=self.model._x),
            entry('_p_est_prev', struct=self._p_est_prev),
            entry('_p_set', struct=self._p_set),
            entry('_tvp', repeat=self.n_horizon, struct=self.model._tvp),
            entry('_y_meas', repeat=self.n_horizon, struct=self.model._y),
        ])
        self.n_opt_p = opt_p.shape[0]

        # Dummy struct with symbolic variables
        self.aux_struct = self.model.sv.sym_struct([
            entry('_aux', repeat=[self.n_horizon], struct=self.model._aux_expression)
        ])
        # Create mutable symbolic expression from the struct defined above.
        self._opt_aux = opt_aux = self.model.sv.struct(self.aux_struct)

        self.n_opt_aux = opt_aux.shape[0]

        self.lb_opt_x = opt_x(-np.inf)
        self.ub_opt_x = opt_x(np.inf)

        # Initialize objective function and constraints
        obj = DM(0)
        cons = []
        cons_lb = []
        cons_ub = []

        # Arrival cost:
        arrival_cost = self.arrival_cost_fun(
            opt_x_unscaled['_x', 0, -1],
            opt_p['_x_prev'],#/self._x_scaling,
            opt_x_unscaled['_p_est'],
            opt_p['_p_est_prev'],#/self._p_est_scaling
            opt_p['_p_set']
            )

        obj += arrival_cost

        # Get concatenated parameters vector containing the estimated and fixed parameters (scaled)
        _p = self._p_cat_fun(opt_x['_p_est'], opt_p['_p_set']/self._p_set_scaling)

        # For all control intervals
        for k in range(self.n_horizon):
            # Compute constraints and predicted next state of the discretization scheme
            col_xk = vertcat(*opt_x['_x', k+1, :-1])
            col_zk = vertcat(*opt_x['_z', k])
            [g_ksb, xf_ksb] = ifcn(opt_x['_x', k, -1], col_xk,
                                   opt_x['_u', k], col_zk, opt_p['_tvp', k],
                                   _p, opt_x['_w', k])

            # Compute current measurement
            yk_calc = self.model._meas_fun(opt_x_unscaled['_x', k+1, -1], opt_x_unscaled['_u', k], opt_x_unscaled['_z', k, 0],
                opt_p['_tvp', k], _p, opt_x_unscaled['_v', k])

            # Add the collocation equations
            cons.append(g_ksb)
            cons_lb.append(np.zeros(g_ksb.shape[0]))
            cons_ub.append(np.zeros(g_ksb.shape[0]))

            # Add continuity constraints
            cons.append(xf_ksb - opt_x['_x', k+1, -1])
            cons_lb.append(np.zeros((self.model.n_x, 1)))
            cons_ub.append(np.zeros((self.model.n_x, 1)))

            # Add measurement constraints
            cons.append(yk_calc - opt_p['_y_meas', k])
            cons_lb.append(np.zeros((self.model.n_y, 1)))
            cons_ub.append(np.zeros((self.model.n_y, 1)))

            k_eps = min(k, n_eps-1)
            if self.nl_cons_check_colloc_points:
                # Ensure nonlinear constraints on all collocation points
                for i in range(n_total_coll_points):
                    nl_cons_k = self._nl_cons_fun(
                        opt_x_unscaled['_x', k, i], opt_x_unscaled['_u', k], opt_x_unscaled['_z', k, i],
                        opt_p['_tvp', k], opt_x['_p_est'], opt_p['_p_set'], opt_x_unscaled['_eps', k_eps])
                    cons.append(nl_cons_k)
                    cons_lb.append(self._nl_cons_lb)
                    cons_ub.append(self._nl_cons_ub)
            else:
                # Ensure nonlinear constraints only on the beginning of the FE
                nl_cons_k = self._nl_cons_fun(
                    opt_x_unscaled['_x', k, -1], opt_x_unscaled['_u', k], opt_x_unscaled['_z', k, 0],
                    opt_p['_tvp', k], opt_x['_p_est'], opt_p['_p_set'], opt_x_unscaled['_eps', k_eps])
                cons.append(nl_cons_k)
                cons_lb.append(self._nl_cons_lb)
                cons_ub.append(self._nl_cons_ub)

            cons.append(nl_cons_k)
            cons_lb.append(self._nl_cons_lb)
            cons_ub.append(self._nl_cons_ub)


            obj += self.stage_cost_fun(
                opt_x_unscaled['_w', k], opt_x_unscaled['_v', k], opt_p['_tvp', k], _p
            )

            # Add slack variables to the cost
            obj += self.epsterm_fun(opt_x_unscaled['_eps', k_eps])


            # Calculate the auxiliary expressions for the current scenario:
            opt_aux['_aux', k] = self.model._aux_expression_fun(
                opt_x_unscaled['_x', k, -1], opt_x_unscaled['_u', k], opt_x_unscaled['_z', k, -1], opt_p['_tvp', k], _p)


        if self.cons_check_colloc_points:   # Constraints for all collocation points.
            # Bounds for the states on all discretize values along the horizon
            self.lb_opt_x['_x'] = self._x_lb.cat/self._x_scaling
            self.ub_opt_x['_x'] = self._x_ub.cat/self._x_scaling

            # Bounds for the algebraic states along the horizon
            self.lb_opt_x['_z'] = self._z_lb.cat/self._z_scaling
            self.ub_opt_x['_z'] = self._z_ub.cat/self._z_scaling
        else:   # Constraints only at the beginning of the finite Element
            # Bounds for the states on all discretize values along the horizon
            self.lb_opt_x['_x', 1:self.n_horizon, -1] = self._x_lb.cat/self._x_scaling
            self.ub_opt_x['_x', 1:self.n_horizon, -1] = self._x_ub.cat/self._x_scaling

            # Bounds for the algebraic states along the horizon
            self.lb_opt_x['_z', :, 0] = self._z_lb.cat/self._z_scaling
            self.ub_opt_x['_z', :, 0] = self._z_ub.cat/self._z_scaling

        # Bounds for the inputs along the horizon
        self.lb_opt_x['_u'] = self._u_lb.cat/self._u_scaling
        self.ub_opt_x['_u'] = self._u_ub.cat/self._u_scaling

        # Bounds for the slack variables along the horizon:
        self.lb_opt_x['_eps'] = self._eps_lb.cat
        self.ub_opt_x['_eps'] = self._eps_ub.cat

        # Bounds for the inputs along the horizon
        self.lb_opt_x['_p_est'] = self._p_est_lb.cat/self._p_est_scaling
        self.ub_opt_x['_p_est'] = self._p_est_ub.cat/self._p_est_scaling


        # Write all created elements to self:
        self._nlp_obj = obj
        self._nlp_cons = cons
        self._nlp_cons_lb = cons_lb
        self._nlp_cons_ub = cons_ub

        # Initialize copies of structures with numerical values (all zero):
        self._opt_x_num = self._opt_x(0)
        self.opt_x_num_unscaled = self._opt_x(0)
        self._opt_p_num = self._opt_p(0)
        self.opt_aux_num = self._opt_aux(0)

        self.flags['prepare_nlp'] = True


    def _create_nlp(self):
        """Internal method. See detailed documentation in optimizer.create_nlp
        """


        self._nlp_cons = vertcat(*self._nlp_cons)
        self._nlp_cons_lb = vertcat(*self._nlp_cons_lb)
        self._nlp_cons_ub = vertcat(*self._nlp_cons_ub)


        # Validity check:
        _test_obj_fun = Function('f', [self._opt_x, self._opt_p], [self._nlp_obj])
        _test_cons_fun = Function('f', [self._opt_x, self._opt_p], [self._nlp_cons])
        try:
            _test_obj_fun(self.opt_x_num,self.opt_p_num)
        except:
            err_msg = 'The MHE optimization problem objective function contains unknown symbolic variables.'
            raise Exception(err_msg)
        try:
            _test_cons_fun(self.opt_x_num,self.opt_p_num)
        except:
            err_msg = 'The MHE optimization problem constraint function contains unknown symbolic variables.'
            raise Exception(err_msg)


        self.n_opt_lagr = self._nlp_cons.shape[0]
        # Create casadi optimization object:
        nlpsol_opts = {
            'expand': False,
            'ipopt.linear_solver': 'mumps',
        }.update(self.nlpsol_opts)
        nlp = {'x': vertcat(self._opt_x), 'f': self._nlp_obj, 'g': self._nlp_cons, 'p': vertcat(self._opt_p)}
        self.S = nlpsol('S', 'ipopt', nlp, self.nlpsol_opts)



        # Create function to caculate all auxiliary expressions:
        self.opt_aux_expression_fun = Function('opt_aux_expression_fun', [self._opt_x, self._opt_p], [self._opt_aux])

        # Gather meta information:
        meta_data = {key: getattr(self, key) for key in self.data_fields}
        self.data.set_meta(**meta_data)

        self._prepare_data()
        self.flags['setup'] = True
