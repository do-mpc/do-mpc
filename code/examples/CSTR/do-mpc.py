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
import sys
sys.path.insert(0,path_do_mpc+'setup_functions')
sys.path.insert(0,path_do_mpc+'aux_functions')
sys.dont_write_bytecode = True
# Start CasADi
from casadi import *
# Import do-mpc
import core_do_mpc
# Import do-mpc plotting routines
import plotting_routines
import setup_solver

"""
==========================================================================
DO-MPC: Definition of the problem
--------------------------------------------------------------------------
        Modify the following templates according to your problem
==========================================================================
"""

# Import the user defined modules
import template_model
import template_optimizer
import template_observer
import template_simulator

# Create the objects for each module
model_1 = template_model.model()
# Create an optimizer object based on the template and a model
optimizer_1 = template_optimizer.optimizer(model_1)
# Create an observer object based on the template and a model
observer_1 = template_observer.observer(model_1)
# Create a simulator object based on the template and a model
simulator_1 = template_simulator.simulator(model_1)
# Create a configuration
configuration_1 = core_do_mpc.configuration(model_1, optimizer_1, observer_1, simulator_1)


"""
==========================================================================
DO-MPC: Set up of the optimization problem
--------------------------------------------------------------------------
        For a standard use of DO-MPC it is not necessary to modify the
        following files
==========================================================================
"""

# Setup the solver for the given configuration
configuration_1 = core_do_mpc.setup_solver(configuration_1)
# Initialize the data structures
mpc_data = core_do_mpc.mpc_data(configuration_1)


"""
==========================================================================
DO-MPC: Solution of the MPC problem
--------------------------------------------------------------------------
        This part of the code implements a standard MPC loop
==========================================================================
"""
while (configuration_1.simulator.t0_sim < configuration_1.optimizer.t_end - 0.005):

    """
    ==========================================================================
    DO-MPC: Optimizer
    ==========================================================================
    """
    # Make one optimizer step (solve the NLP)
    configuration_1.optimizer.make_step()

    """
    ==========================================================================
    DO-MPC: Simulator
    ==========================================================================
    """
    # Simulate the system one step using the solution obtained in the optimization
    configuration_1.simulator.make_step(configuration_1)

    """
    ==========================================================================
    DO-MPC: Observer
    ==========================================================================
    """
    # Make one observer step
    configuration_1.observer.make_step(configuration_1)
    """
    ==========================================================================
    DO-MPC: Prepare next iteration and store information
    ==========================================================================
    """
    # Store the information
    mpc_data.store_step(configuration_1)
    # Set initial condition constraint for the next iteration
    configuration_1.optimizer.prepare_next(configuration_1)

    # Plot animation
    if configuration_1.simulator.plot_anim:
        plotting_routines.plot_animation(configuration_1, mpc_data)
    	raw_input("Press Enter to continue...")


"""
==========================================================================
DO-MPC: Plot the results
==========================================================================
"""

plotting_routines.plot_mpc(configuration_1, mpc_data)

# Export to matlab if wanted
if configuration_1.simulator.export_to_matlab:
    execfile(path_do_mpc+"aux_functions/export_to_matlab.py")
    export_to_matlab(mpc_states, mpc_control, mpc_time, mpc_cpu, index_mpc, x_scaling, u_scaling, export_name)
    print("Exporting to Matlab as ''" + export_name + "''")

raw_input("Press Enter to exit DO-MPC...")
