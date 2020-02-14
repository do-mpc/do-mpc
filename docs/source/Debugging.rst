Debugging
=========


Feasibility problems
********************

.. note::
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



       

