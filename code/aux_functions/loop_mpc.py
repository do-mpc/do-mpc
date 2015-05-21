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

"""
This file contains the definition of the functions used in the mpc loop
"""
from casadi import *
import numpy as NP

def initialize_first_iteration(end_time, t_step, nx, nu, np, x0, u0):
	index_mpc = 1
	mpc_states = NP.resize(NP.array([]),(end_time/t_step + 1,nx))
	mpc_control= NP.resize(NP.array([]),(end_time/t_step + 1 ,nu))
	mpc_alg	   = NP.resize(NP.array([]),(end_time/t_step + 1,1))
	mpc_time   = NP.resize(NP.array([]),(end_time/t_step + 1))
	mpc_cost   = NP.resize(NP.array([]),(end_time/t_step + 1))
	mpc_ref   = NP.resize(NP.array([]),(end_time/t_step + 1))
	mpc_cpu    = NP.resize(NP.array([]),(end_time/t_step + 1))
	mpc_parameters = NP.resize(NP.array([]),(end_time/t_step + 1,np))
	mpc_states[0,:] = x0
	mpc_control[0,:]= u0
	mpc_time[0]		= 0
	x0_sim=x0
	
	return mpc_states, mpc_control, mpc_alg, mpc_time, mpc_cost, mpc_cpu, mpc_parameters, x0_sim
	
def loop_optimizer(solver):
	solver.evaluate()
	optimal_cost = NP.array(solver.output(NLP_SOLVER_F))
	constraints = NP.array(solver.output(NLP_SOLVER_G))
	optimal_solution = NP.array(solver.output(NLP_SOLVER_X))
	
	return optimal_solution, optimal_cost, constraints
	
def loop_simulator(x0_sim, u_mpc, p_real, simulator, t0_sim, t_step):
	tf_sim = t0_sim + t_step
	simulator.setOption("t0",t0_sim)
	simulator.setOption("tf",tf_sim)
	simulator.setInput(vertcat((u_mpc,p_real)),INTEGRATOR_P)
	simulator.setInput(x0_sim,INTEGRATOR_X0)
	simulator.evaluate()	 
	xf_sim = NP.squeeze(simulator.output())
	return xf_sim
	
def loop_measure(xf_sim):
	y_meas = deepcopy(xf_sim)
	#Introduce here noise if the measurement contains noise
	return y_meas
	
def loop_observer(y):
	x = template_observer(y)
	return x

def loop_initialize(solver, x0_meas, u_mpc, v_opt, vars_lb, vars_ub, X_offset, U_offset, nx, nu, t0_sim, t_step, index_mpc):
	# Set initial condition constraint for the next iteration
	vars_lb[X_offset[0,0]:X_offset[0,0]+nx] = x0_meas
	vars_ub[X_offset[0,0]:X_offset[0,0]+nx] = x0_meas
	solver.setInput(vars_lb,NLP_SOLVER_LBX)
	solver.setInput(vars_ub,NLP_SOLVER_UBX)
	# Set the last control input as parameter for the penalty term
	solver.setInput(u_mpc, NLP_SOLVER_P)
	# Set current solution as initial guess for next iteration
	solver.setInput(v_opt,NLP_SOLVER_X0)
	solver.setInput(solver.output(NLP_SOLVER_LAM_X),NLP_SOLVER_LAM_X0)
	solver.setInput(solver.output(NLP_SOLVER_LAM_G),NLP_SOLVER_LAM_G0)
	t0_sim = t0_sim + t_step
	index_mpc = index_mpc + 1
	return solver, vars_lb, vars_ub, t0_sim, index_mpc

def loop_store(x0_sim, u_mpc, t0_sim, t_step, p_real, index_mpc, solver, mpc_states, mpc_control, mpc_time, mpc_parameters, mpc_cpu):
	mpc_states[index_mpc,:] = x0_sim
	mpc_control[index_mpc,:] = u_mpc
	mpc_time[index_mpc] = t0_sim + t_step
	mpc_parameters[index_mpc,:] = p_real
	aux = solver.getStats()
	mpc_cpu[index_mpc] = aux['t_mainloop']	
	return mpc_states, mpc_control, mpc_time, mpc_cpu
