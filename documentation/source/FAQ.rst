***
FAQ
***
Some tips and tricks when you can't rule them all.

Time-varying parameters
#######################
Time-varying parameters are an important feature of **do-mpc**.
But when do I need them, how are they implemented and what makes them different from regular parameters?

With model predictive control and moving horizon estimation we are considering finite future (control) or past (estimation) trajectories
based on a model of our system.
These finite sequences are shifting at each estimation and control step.
**Time-varying parameters** are required, when:

* the model is subject to some exterior influence (e.g. weather prediction) that is varying at each element of the sequence.

* the MPC/MHE cost function contains elements (e.g. a reference for control) that is varying at each element of the sequence.

Both cases have in common that the parameters are **a priori known** and not constant over the prediction / estimation horizon.
This is the main difference to regular **parameters** which typically only influence the model (not the cost function)
and can be estimated with moving horizon estimation and considered as parametric uncertainties for robust model predictive control.


Implementation
**************

Time-varying parameters are always introduced in the **do-mpc** :py:class:`do_mpc.model.Model` with the
:py:mod:`do_mpc.model.Model.set_variable` method. For example:

::

	model_type = 'continuous' # either 'discrete' or 'continuous'
	model = do_mpc.model.Model(model_type)

	# Introduce state temperature:
	temperature = model.set_variable(var_type='_x', var_name='temperature')
	# Introduce tvp: Set-point for the temperature
	temperature_set_point= model.set_variable(var_type='_tvp', var_name='temperature_set_point')
	# Introduce tvp: External temperature (disturbance)
	temperature_external = model.set_variable(var_type='_tvp', var_name='temperature_external')

	...

The obtained time-varying parameters can be used throughout the model and all derived classes.
In the shown example, we assume that the external temperature has an influence on our temperature state.
We can thus incorporate this variable in the ODE:

::

	model.set_rhs('temperature', alpha*(temperature_external-temperature))


MPC configuration
-----------------

Furthermore, we want to use the introduced set-point in a quadratic MPC cost function.
To do this, we initiate an :py:class:`do_mpc.controller.MPC` object with the configured model:

::

	mpc = do_mpc.controller.MPC(model)

	mpc.set_param(n_horizon = 20, t_step = 60)

And then use the attributes :py:attr:`do_mpc.model.Model.x` and :py:attr:`do_mpc.model.Model.tvp`
to formulate a quadratic tracking cost.

::

	lterm = (model.x['temperature']-model.tvp['temperature_set_point'])**2
	mterm = lterm
	mpc.set_objective(lterm=lterm, mterm=mterm)

.. note::

	We assume here that the ``mpc`` controller is not configured in the same Python scope as the ``model``.
	Thus the variables (e.g. ``temperature_external``, ``temperature``, ...) are not necessarily available.
	Instead, these variables are obtained from the model with the shown attributes.

After invoking the :py:meth:`do_mpc.controller.MPC.setup` method this will create the following cost function:

.. math::

	J = \sum_{k=0}^{N+1} (T_k-T_{k,\text{set}})^2

The only problem that remains is: What are the values for the set-point for the temperature and the external temperature for the ODE equation?
So far we have only introduced them as symbolic variables.

What makes the definition of these values so complicated is that at each control step, we need not only a single value for
these variables but an entire sequence.
Furthermore, these sequences are not necessarily the same (shifted) values at the next step.

To address this problem **do-mpc** allows the user to declare a **tvp-function** with :py:meth:`do_mpc.controller.MPC.set_tvp_fun`
which is internally invoked at each call of the MPC controller with :py:meth:`do_mpc.controller.MPC.make_step`.

The **tvp-function** returns numerical values for the currently valid sequences and passes them to the optimizer.
Because the **tvp-function** is user-defined, the approach allows for the greatest flexibility.

**do-mpc** also ensures that the output of this function is consistent with the configuration of the model and controller.
This is achieved by requiring the output of the **tvp-function** to be of a particular structure which can be obtained with
:py:meth:`do_mpc.controller.MPC.get_tvp_template`. This structure can be indexed with a time-step and the name of
a previously introduced time-varying parameter. Through indexing these values can be obtained and set conveniently.

In the following we show how this works in practice. The first step is to obtain the ``tvp_template``:

::

	tvp_template = mpc.get_tvp_template()


Afterwards, we define a function that takes as input the current time and returns the ``tvp_template``
filled with the currently valid sequences.

::

	def tvp_fun(t_now):
		for k in range(n_horizon+1):
			tvp_template['_tvp',k,'temperature_set_point'] = 10
			tvp_template['_tvp',k,'temperature_external'] = 20

		return tvp_template

.. note::

	Within the ``tvp_fun`` above, the user is free to perform any operation.
	Typically, the data for the time-varying parameters is read from a numpy array or obtained as a function of the current time.


The function ``tvp_fun`` can now be treated similarly to a variable in the current python scope.
The final step of the process is to pass this function with :py:meth:`do_mpc.controller.MPC.set_tvp_fun`:

::

	mpc.set_tvp_fun(tvp_fun)

The configuration of the MPC controller is thus completed.



MHE configuration
-----------------

The MHE configuration of the time-varying parameters is equivalent to the MPC configuration shown above.

Simulator configuration
-----------------------

The simulator also needs to be adapted for time-varying parameters
because we cannot evaluate the previously introduced ODE without a numerical value for
``temperature_external``.

The logic is the same as for the MPC controller and MHE estimator: We get the ``tvp_template`` with :py:meth:`do_mpc.simulator.Simulator.get_tvp_template`
define a function ``tvp_fun`` and pass it to the simulator with :py:meth:`do_mpc.simulator.Simulator.set_tvp_fun`

The configuration of the simulator is significantly easier however,
because we only need a single value of this parameter instead of a sequence:

::

	# Get simulator instance. The model contains _tvp.
	simulator = do_mpc.simulator.Simulator(model)
	# Set some required parameters
	simulator.set_param(t_step = 60)

	# Get the template
	tvp_template = simulator.get_tvp_template()

	# Define the function (indexing is much simpler ...)
	def tvp_fun(t_now):
		tvp_template['temperature_external'] = ...
		return tvp_template

	# Set the tvp_fun:
	simulator.set_tvp_fun(tvp_fun)

.. note::

	All time-varying parameters that are not explicitly set default to ``0`` in the ``tvp_template``.
	Thus, if some parameters are not required (e.g. they were introduced for the controller),
	they don't need to be set in the ``tvp_fun``. This is shown here, where the simulator doesn't need the set-point.

.. note::

	From the perspective of the simulator there is no difference between time-varying parameters (``_tvp``) and regular parameters (``_p``).
	The difference is important only for the MPC controller and MHE estimator.
	These methods consider a finite sequence of future / past information, e.g. the weather, which can change over time.
	Parameters, on the other hand, are constant over the entire horizon.


Feasibility issues
##################
A common problem with MPC control and MHE estimation are feasibility issues
that arise when the solver cannot satisfy the constraints of the optimization problem.


Is the initial state feasible?
******************************
With MPC, a problem is infeasible if the initial state is infeasible.
This can happen in the close-loop application, where the state prediction
may vary from the true state evolution.
The following tips may be used to diagnose and fix this (and other) problems.

Which constraints are violated?
*******************************
Check which bound constraints are violated. Retrieve the (infeasible) "optimal" solution and compare it to the bounds:

::

	lb_bound_violation = mpc.opt_x_num.cat <= mpc.lb_opt_x
	ub_bound_violation = mpc.opt_x_num.cat <= mpc.ub_opt_x

Retrieve the labels from the optimization variables and find those that are violating the constraints:

::

	opt_labels = mpc.opt_x.labels()
	labels_lb_viol =np.array(opt_labels)[np.where(lb_viol)[0]]
	labels_ub_viol =np.array(opt_labels)[np.where(lb_viol)[0]]

The arrays ``labels_lb_viol`` and ``labels_ub_viol`` indicate which variables are problematic.


Use soft-constraints.
*********************

Some control problems, especially with economic objective will lead to trajectories operating close to (some) constraints.
Uncertainty or model inaccuracy may lead to constraint violations and thus infeasible (usually nonsense) solutions.
Using soft-constraints may help in this case.
Both the MPC controller and MHE estimator support this feature, which can be configured with (example for MPC):

::

	mpc.set_nl_cons('cons_name', expression, upper_bound, soft_constraint=True)

See the full feature documentation here: :py:mod:`do_mpc.optimizer.Optimizer.set_nl_cons`


Silence IPOPT
#############

IPOPT is the default solver for the :py:class:`do_mpc.controller.MPC` controller and :py:class:`do_mpc.estimator.MHE` estimator.
While we generally **recommend to have a look at the solver output**, to check for feasibility issues,
it may be useful to silence IPOPT in some cases.

This can be achieved conveniently over the :py:class:`do_mpc.controller.MPCSettings` and :py:class:`do_mpc.estimator.MHESettings` 
which are stored as the attribute ``settings``, e.g.

::

    # for the MPC

    mpc.settings.supress_ipopt_output()

    # or for the MHE

    mhe.settings.supress_ipopt_output()