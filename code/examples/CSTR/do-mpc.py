# 	 -*- coding: utf-8 -*-
#
#    This file is part of DO-MPC
#
#    DO-MPC: An environment for the easy, modular and efficient implementation of
#            robust nonlinear model predictive control
#
#    The MIT License (MIT)
#
#    Copyright (c) 2014-2015 Sergio Lucia, Alexandru Tatulea-Codrean, Sebastian Engell
#                            TU Dortmund. All rights reserved
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
#

# This is the main path of your DO-MPC installation relative to the execution folder
path_do_mpc = '../../'

import core_do_mpc
"""
==========================================================================
DO-MPC: Import Commands
--------------------------------------------------------------------------
        Add any desired library here
==========================================================================
"""
execfile(path_do_mpc+"aux_functions/import_modules.py")

"""
==========================================================================
DO-MPC: Definition of the problem
--------------------------------------------------------------------------
        Modify the following templates according to your problem
==========================================================================
"""

# Define the model and the OCP
execfile("template_model.py")

# Define the optimizer
execfile("template_optimizer.py")

# Define the observer
execfile("template_observer.py")

# Define the simulator
execfile("template_simulator.py")

"""
==========================================================================
DO-MPC: Set up of the optimization problem
--------------------------------------------------------------------------
        For a standard use of DO-MPC it is not necessary to modify the
        following files
==========================================================================
"""
# Set up the NLP based on collocation or multiple-shooting
execfile(path_do_mpc+"setup_functions/setup_nlp.py")

# Set up the solver
execfile(path_do_mpc+"setup_functions/setup_solver.py")

# Set up plotting functions and other auxiliary functions
execfile(path_do_mpc+"aux_functions/setup_aux.py")
execfile(path_do_mpc+"aux_functions/loop_mpc.py")

# First formulate the optimization problem usign all the previous information
solver, X_offset, U_offset, E_offset, vars_lb, vars_ub, end_time, t_step, x0, u0, x, u, p, x_scaling, u_scaling, nk, parent_scenario, child_scenario, n_branches, n_scenarios, arg = setup_solver()
# Define the simulator of the real system
simulator = template_simulator(t_step)
# Initialize necessary variables
current_time = 0
t0_sim = current_time
tf_sim = current_time + t_step
index_mpc = 1
nx = x0.size
nu = u0.size
np = p.size(1)
# Initialize MPC information structures
mpc_states, mpc_control, mpc_alg, mpc_time, mpc_cost, mpc_cpu, mpc_parameters, x0_sim = initialize_first_iteration(end_time, t_step, nx, nu, np, x0, u0)

"""
==========================================================================
DO-MPC: Solution of the MPC problem
--------------------------------------------------------------------------
        This part of the code implements a standard MPC loop
==========================================================================
"""

# load the plotting options defined in template_simulator.py
plot_states, plot_control, plot_anim, export_to_matlab, export_name = plotting_options()

while (current_time < end_time):
	# Define the real value of the uncertain parameters used for the simulation
	# It can be changed as a function of time
	p_real = real_parameters(current_time)
	"""
	==========================================================================
	DO-MPC: Optimizer
	==========================================================================
	"""
	# Solve the NLP
	optimal_solution, optimal_cost, constraints = loop_optimizer(solver, arg)

	# Extract the control input that will be injected to the plant
	v_opt = optimal_solution;
	u_mpc = NP.squeeze(v_opt[U_offset[0][0]:U_offset[0][0]+nu])
	"""
	==========================================================================
	DO-MPC: Simulator
	==========================================================================
	"""
	xf_sim = loop_simulator(x0_sim, u_mpc, p_real, simulator, t0_sim, t_step);
	x0_sim = xf_sim

	"""
	==========================================================================
	DO-MPC: Observer
	==========================================================================
	"""
	# Measurement and observer
	y_meas = loop_measure(xf_sim)
	xf_meas = loop_observer(y_meas)

	"""
	==========================================================================
	DO-MPC: Prepare next iteration and store information
	==========================================================================
	"""
	# Store the infromation
	mpc_states, mpc_control, mpc_time, mpc_cpu = loop_store(x0_sim, u_mpc, t0_sim, t_step, p_real, index_mpc, solver, mpc_states, mpc_control, mpc_time, mpc_parameters, mpc_cpu)
	# Set initial condition constraint for the next iteration
	solver, vars_lb, vars_ub, t0_sim, index_mpc, arg = loop_initialize(solver, xf_meas, u_mpc, v_opt, vars_lb, vars_ub, X_offset, U_offset, nx, nu, t0_sim, t_step, index_mpc, arg)
	current_time = t0_sim;
	# Plot animation

	if plot_anim:
		plot_animation(mpc_states, mpc_control, mpc_time, index_mpc, plot_states, plot_control, x_scaling, u_scaling, x, u, X_offset, U_offset, n_branches, n_scenarios, nk, t0_sim-t_step, child_scenario, parent_scenario, t_step, v_opt)
		raw_input("Press Enter to continue...")


"""
==========================================================================
DO-MPC: Plot the results
==========================================================================
"""
plot_mpc(mpc_states, mpc_control, mpc_time, index_mpc, plot_states, plot_control, x_scaling, u_scaling, x, u)
if export_to_matlab:
    execfile(path_do_mpc+"aux_functions/export_to_matlab.py")
    export_to_matlab(mpc_states, mpc_control, mpc_time, mpc_cpu, index_mpc, x_scaling, u_scaling, export_name)
    print("Exporting to Matlab as ''" + export_name + "''")

raw_input("Press Enter to exit DO-MPC...")
