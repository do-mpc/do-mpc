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

def plot_mpc(mpc_states, mpc_control, mpc_time, index_mpc, plot_states, plot_control, x_scaling, u_scaling, x, u):
	# This function plots the states and controls chosen in the variables plot_states and plot_control until a certain index (index_mpc)
	plt.ion()
	fig = plt.figure(1)
	total_subplots = len(plot_states) + len(plot_control)
	# First plot the states
	for index in range(len(plot_states)):
		plot = plt.subplot(total_subplots, 1, index + 1)
		plt.plot(mpc_time[0:index_mpc], mpc_states[0:index_mpc,plot_states[index]] * x_scaling[plot_states[index]])
		plt.ylabel(str(x[plot_states[index]]))
		plt.xlabel("Time")
		plt.grid()
		plot.yaxis.set_major_locator(MaxNLocator(4))

	# Plot the control inputs
	for index in range(len(plot_control)):
		plot = plt.subplot(total_subplots, 1, len(plot_states) + index + 1)
		plt.plot(mpc_time[0:index_mpc], mpc_control[0:index_mpc,plot_control[index]] * u_scaling[plot_control[index]] ,drawstyle='steps')
		plt.ylabel(str(u[plot_states[index]]))
		plt.xlabel("Time")
		plt.grid()
		plot.yaxis.set_major_locator(MaxNLocator(4))
			
		

def plot_state_pred(v,t0,el,lineop, n_scenarios, n_brances, nk, child_scenario, X_offset, x_scaling, t_step):
  # This function plots the prediction of a state
  #plt.clf()
  plt.hold(True)
  # Time grid
  tf = t_step * nk
  tgrid = NP.linspace(t0,t0+tf,nk+1)
  # For all control intervals
  for k in range(nk):
    # For all scenarios
    for s in range(n_scenarios[k]):
      # For all uncertainty realizations
      for b in range(n_branches[k]):
        # Get state trajectory segment
        x_beginning = v[el+X_offset[k][s]]
        s_next = child_scenario[k][s][b]
        x_end = v[el+X_offset[k+1][s_next]]
        x_segment = NP.array([x_beginning,x_end])*x_scaling[el]
        plt.plot(tgrid[k:k+2],x_segment,lineop)			
		

def plot_control_pred(v,t0,el,lineop, n_scenarios, n_brances, nk, parent_scenario, U_offset, u_scaling, t_step, u_last_step):
	# This function plots the prediction of a control input
	plt.hold(True)
	# Time grid
	tf = t_step * nk
	tgrid = NP.linspace(t0,t0+tf,nk+1)
	# For all control intervals
	for k in range(nk):
		# For all scenarios
		for s in range(n_scenarios[k]):
			# Time segment
			t_beginning = tgrid[k]
			t_end = tgrid[k+1]

			# Plot state trajectory segment
			u_this = v[el+U_offset[k][s]]*u_scaling[el]
			plt.plot(NP.array([t_beginning,t_end]),NP.array([u_this,u_this]),lineop)

			# Plot vertical line connecting the scenarios
			if k == 0:
				u_prev = u_last_step
			else:
				u_prev = v[el+U_offset[k-1][parent_scenario[k][s]]]*u_scaling[el]
			plt.plot(NP.array([t_beginning,t_beginning]),NP.array([u_prev,u_this]),lineop)


def plot_animation(mpc_states, mpc_control, mpc_time, index_mpc, plot_states, plot_control, x_scaling, u_scaling, x, u, X_offset, U_offset, n_branches, n_scenarios, nk, t0, child_scenario, parent_scenario, t_step, v_opt):
	plt.ion()
	total_subplots = len(plot_states) + len(plot_control)
	plt.figure(2)
	# Clear the previous animation
	plt.clf()
	# First plot the states
	for index in range(len(plot_states)):
		plot = plt.subplot(total_subplots, 1, index + 1)
		# First plot the prediction
		plot_state_pred(v_opt, t0, plot_states[index], '-b', n_scenarios, n_branches, nk, child_scenario, X_offset, x_scaling, t_step)
		plt.plot(mpc_time[0:index_mpc-1], mpc_states[0:index_mpc-1,plot_states[index]] * x_scaling[plot_states[index]], '-k', linewidth=2.0)
		plt.ylabel(str(x[plot_states[index]]))
		plt.xlabel("Time")
		plt.grid()
		plot.yaxis.set_major_locator(MaxNLocator(4))

	# Plot the control inputs
	for index in range(len(plot_control)):
		plot = plt.subplot(total_subplots, 1, len(plot_states) + index + 1)
		# First plot the prediction
		plot_control_pred(v_opt, t0, plot_control[index], '-b', n_scenarios, n_branches, nk, parent_scenario, U_offset, u_scaling, t_step, mpc_control[index_mpc-2,plot_control[index]])
		plt.plot(mpc_time[0:index_mpc-1], mpc_control[0:index_mpc-1,plot_control[index]] * u_scaling[plot_control[index]],'-k' ,drawstyle='steps', linewidth=2.0)
		plt.ylabel(str(u[plot_control[index]]))
		plt.xlabel("Time")
		plt.grid()
		plot.yaxis.set_major_locator(MaxNLocator(4))
