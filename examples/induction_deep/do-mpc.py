#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2016 Sergio Lucia, Alexandru Tatulea-Codrean
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
#

# This is the main path of your do-mpc installation relative to the execution folder
path_do_mpc = '../../'
# Add do-mpc path to the current directory
import sys
sys.path.insert(0,path_do_mpc+'code')
# Do not write bytecode to maintain clean directories
sys.dont_write_bytecode = True

# Start CasADi
from casadi import *
from casadi.tools import *
# Import do-mpc core functionalities
import core_do_mpc
# Import do-mpc plotting and data managament functions
import data_do_mpc
import numpy as NP
import pdb
"""
-----------------------------------------------
do-mpc: Definition of the do-mpc configuration
-----------------------------------------------
"""

# Import the user defined modules
import template_model
import template_model_sim
import template_optimizer
import template_observer
import template_simulator

# Create the objects for each module
model_1 = template_model.model()
# model_1_sim = template_model_sim.model()
# Create an optimizer object based on the template and a model
optimizer_1 = template_optimizer.optimizer(model_1)
# Create an observer object based on the template and a model
observer_1 = template_observer.observer(model_1)
# Create a simulator object based on the template and a model
simulator_1 = template_simulator.simulator(model_1)
# Create a configuration
configuration_1 = core_do_mpc.configuration(model_1, optimizer_1, observer_1, simulator_1)

# Set up the solvers
configuration_1.setup_solver()
index_mpc = 0
index_stationary = 0

"""
----------------------------
do-mpc: MPC loop
----------------------------
"""
while (configuration_1.simulator.t0_sim + configuration_1.simulator.t_step_simulator < configuration_1.optimizer.t_end):

    """
    ----------------------------
    do-mpc: Optimizer
    ----------------------------
    """
    # Always reinitialize the "time" state for the optimization
    X_offset = configuration_1.optimizer.nlp_dict_out['X_offset']
    # configuration_1.optimizer.arg['lbx'][X_offset[0,0]+ 2] = 0
    # configuration_1.optimizer.arg['ubx'][X_offset[0,0]+ 2] = 0
    # Make one optimizer step (solve the NLP)
    # if configuration_1.simulator.t0_sim == 0:
    if index_stationary == 0 or index_stationary >= 0:
        configuration_1.make_step_optimizer()
    if index_stationary < 0:
        configuration_1.optimizer.u_mpc = NP.array([0.5,60])
        index_stationary += 1

    """
    ----------------------------
    do-mpc: Simulator
    ----------------------------
    """
    # Simulate the system one step using the solution obtained in the optimization
    for index_sim in range(int(1/configuration_1.simulator.t_step_simulator)):
        configuration_1.make_step_simulator()

        """
        ----------------------------
        do-mpc: Observer
        ----------------------------
        """
        # Make one observer step
        configuration_1.make_step_observer()

        """
        ------------------------------------------------------
        do-mpc: Prepare next iteration and store information
        ------------------------------------------------------
        """
        # Store the information
        configuration_1.store_mpc_data()

        # Set initial condition constraint for the next iteration
        configuration_1.prepare_next_iter()

        """
        ------------------------------------------------------
        do-mpc: Plot MPC animation if chosen by the user
        ------------------------------------------------------
        """
        # Plot animation if chosen in by the user
        data_do_mpc.plot_animation(configuration_1)

    configuration_1.simulator.x0_sim[2] = 0
    configuration_1.simulator.xf_sim[2] = 0
    configuration_1.optimizer.arg['lbx'][X_offset[0,0]+ 2] = 0
    configuration_1.optimizer.arg['ubx'][X_offset[0,0]+ 2] = 0
    """
    ------------------------------------------------------
    do-mpc: Bias correction term
    ------------------------------------------------------
    """
    BIAS_CORRECTION = 0
    if BIAS_CORRECTION:
        # Update the set-point to track the correct power
        step_index = int(configuration_1.simulator.t0_sim / configuration_1.simulator.t_step_simulator)
        if (index_mpc == 0):
            tv_param_1_adapted = configuration_1.optimizer.tv_p_values[step_index][0]
            tv_param_2_adapted = configuration_1.optimizer.tv_p_values[step_index][1]
            tv_p_values_adapted = configuration_1.optimizer.tv_p_values
        elif (configuration_1.optimizer.tv_p_values[step_index][0,0] != configuration_1.optimizer.tv_p_values[step_index-1][0,0]):
            tv_param_1_adapted = configuration_1.optimizer.tv_p_values[step_index][0]
            tv_param_2_adapted = configuration_1.optimizer.tv_p_values[step_index][1]
            tv_p_values_adapted = configuration_1.optimizer.tv_p_values
        else:
            pass
        # load original tv_param
        tv_p_values_original = configuration_1.optimizer.tv_p_values[step_index]
        # Compute the bias term
        mpc_other = configuration_1.mpc_data.mpc_other
        steps_per_cycle = int(1/configuration_1.simulator.t_step_simulator)
        meas_power = NP.mean(mpc_other[index_mpc*steps_per_cycle:(index_mpc+1)*steps_per_cycle,1])
        index_mpc += 1
        # Assume that the simulated power is the original in the cost (exact tracking)
        sim_power = tv_p_values_original[0]
        # pdb.set_trace()
        bias_term = meas_power - sim_power

        if index_mpc <= 2:
            k_bias =  0.0 # The first steps the power is not correctly computed because of initial cond
        else:
            k_bias = -0.8

        tv_param_1_adapted = tv_param_1_adapted + k_bias * bias_term
        tv_param_2_adapted = tv_param_2_adapted
        tv_p_values_adapted[step_index] = NP.array([tv_param_1_adapted,tv_param_2_adapted])
        # Update in the structure of the controller
        nx = configuration_1.model.ocp.x0.size(1)
        nu = configuration_1.model.u.size(1)
        ntv_p = configuration_1.model.tv_p.size(1)
        nk = configuration_1.optimizer.n_horizon
        parameters_setup_nlp = struct_symMX([entry("uk_prev",shape=(nu)), entry("TV_P",shape=(ntv_p,nk))])
        param = parameters_setup_nlp(0)
        param["TV_P"] = tv_p_values_adapted[step_index]
        param["uk_prev"] = configuration_1.optimizer.u_mpc
        configuration_1.optimizer.arg['p'] = param

"""
------------------------------------------------------
do-mpc: Plot the closed-loop results
------------------------------------------------------
"""

data_do_mpc.plot_mpc(configuration_1)

# Export to matlab if wanted
data_do_mpc.export_to_matlab(configuration_1)


raw_input("Press Enter to exit do-mpc...")
