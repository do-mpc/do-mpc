*********
Debugging
*********
Some tips and tricks when you can't rule them all.

Feasibility problems
####################
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
