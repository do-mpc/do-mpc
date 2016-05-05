# 	 -*- coding: utf-8 -*-
#
#    This file is part of DO-MPC
#
#    DO-MPC: An environment for the easy, modular and efficient implementation of
#            robust nonlinear model predictive control
#
#    The MIT License (MIT)
#
#    Copyright (c) 2014-2015 Sergio Lucia, Alexandru Tatulea-Codrean
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

import setup_nlp
from casadi import *
import numpy as NP

class ocp:
    """ A class that contains a full description of the optimal control problem and will be used in the model class. This is dependent on a specific element of a model class"""
    def __init__(self, param_dict, *opt):
        # Initial state and initial input
        self.x0 = param_dict["x0"]
        self.u0 = param_dict["u0"]
        # Bounds for the states
        self.x_lb = param_dict["x_lb"]
        self.x_ub = param_dict["x_ub"]
        # Bounds for the inputs
        self.u_lb = param_dict["u_lb"]
        self.u_ub = param_dict["u_ub"]
        # Scaling factors
        self.x_scaling = param_dict["x_scaling"]
        self.u_scaling = param_dict["u_scaling"]
        # Symbolic nonlinear constraints
        self.cons = param_dict["cons"]
        # Upper bounds (no lower bounds for nonlinear constraints)
        self.cons_ub = param_dict["cons_ub"]
        # Terminal constraints
        self.cons_terminal = param_dict["cons_terminal"]
        self.cons_terminal_lb = param_dict["cons_terminal_lb"]
        self.cons_terminal_ub = param_dict["cons_terminal_ub"]
        # Flag for soft constraints
        self.soft_constraint = param_dict["soft_constraint"]
        # Penalty term and maximum violation of soft constraints
        self.penalty_term_cons = param_dict["penalty_term_cons"]
        self.maximum_violation = param_dict["maximum_violation"]
        # Lagrange term, Mayer term, and term for input variations
        self.lterm = param_dict["lterm"]
        self.mterm = param_dict["mterm"]
        self.rterm = param_dict["rterm"]

class model:
    """A class for the definition model equations and optimal control problem formulation"""
    def __init__(self, param_dict, *opt):
        # Assert for define length of param_dict
        required_dimension = 24
        if not (len(param_dict) == required_dimension):            raise Exception("Model / OCP information is incomplete. The number of elements in the dictionary is not correct")
        # Assign the main variables describing the model equations
        self.x = param_dict["x"]
        self.u = param_dict["u"]
        self.p = param_dict["p"]
        self.z = param_dict["z"]
        self.rhs = param_dict["rhs"] # Right hand side of the DAE equations
         # Assign the main variables that describe the OCP
        self.ocp = ocp(param_dict)

    @classmethod
    def user_model(cls, param_dict, *opt):
        " This is open for the implementation of a user-defined model class"
        dummy = 1
        return cls(dummy)

class simulator:
    """A class for the definition model equations and optimal control problem formulation"""
    def __init__(self, model_simulator, param_dict, *opt):
        # Assert for define length of param_dict
        required_dimension = 9
        if not (len(param_dict) == required_dimension): raise Exception("Simulator information is incomplete. The number of elements in the dictionary is not correct")
        dae = {'x':model_simulator.x, 'p':vertcat(model_simulator.u,model_simulator.p), 'ode':model_simulator.rhs}
        opts = param_dict["integrator_opts"]
        #FIXME Check the scaling factors!
        #tgrid = linspace(0,t_step,N)
        simulator_do_mpc = integrator("simulator", param_dict["integration_tool"], dae,  opts)
        self.simulator = simulator_do_mpc
        self.plot_states = param_dict["plot_states"]
        self.plot_control = param_dict["plot_control"]
        self.plot_anim = param_dict["plot_anim"]
        self.export_to_matlab = param_dict["export_to_matlab"]
        self.export_name = param_dict["export_name"]
        self.p_real_now = param_dict["p_real_now"]
        self.t_step_simulator = param_dict["t_step_simulator"]
        self.t0_sim = 0
        self.tf_sim = param_dict["t_step_simulator"]
        # TODO: note that here it the same initial condition than for the optimizer is imposed
        self.x0_sim = model_simulator.ocp.x0
        self.xf_sim = 0
        # This is an index to account for the MPC iteration. Starts at 1
        self.mpc_iteration = 1
    @classmethod
    def user_simulator(cls, param_dict, *opt):
        " This is open for the implementation of a user-defined simulator class"
        dummy = 1
        return cls(dummy)

    @classmethod
    def application(cls, param_dict, *opt):
        " This is open for the implementation of connection to a real plant"
        dummy = 1
        return cls(dummy)

    def make_step(self, configuration):
        # Extract the necessary information for the simulation
        u_mpc = configuration.optimizer.u_mpc
        # Use the real parameters
        # TODO: This should allow one to make time-varying changes
        p_real = self.p_real_now(self.t0_sim)
        result  = self.simulator(x0 = self.x0_sim, p = vertcat(u_mpc,p_real))
        self.xf_sim = NP.squeeze(result['xf'])
        # Update the initial condition for the next iteration
        self.x0_sim = self.xf_sim
        # Update the mpc iteration index and the time
        self.mpc_iteration = self.mpc_iteration + 1
        self.t0_sim = self.tf_sim
        self.tf_sim = self.tf_sim + self.t_step_simulator

class optimizer:
    '''This is a class that defines a do-mpc optimizer. The class uses a local model, which
    can be defined independetly from the other modules. The parameters '''
    def __init__(self, optimizer_model, param_dict, *opt):
        # Set the local model to be used by the model
        self.optimizer_model = optimizer_model
        # Assert for the required size of the parameters
        required_dimension = 15
        if not (len(param_dict) == required_dimension): raise Exception("The length of the parameter dictionary is not correct!")
        # Define optimizer parameters
        self.n_horizon = param_dict["n_horizon"]
        self.t_step = param_dict["t_step"]
        self.n_robust = param_dict["n_robust"]
        self.state_discretization = param_dict["state_discretization"]
        self.poly_degree = param_dict["poly_degree"]
        self.collocation = param_dict["collocation"]
        self.n_fin_elem = param_dict["n_fin_elem"]
        self.generate_code = param_dict["generate_code"]
        self.open_loop = param_dict["open_loop"]
        self.t_end = param_dict["t_end"]
        self.nlp_solver = param_dict["nlp_solver"]
        self.linear_solver = param_dict["linear_solver"]
        self.qp_solver = param_dict["qp_solver"]
        # Define model uncertain parameters
        self.uncertainty_values = param_dict["uncertainty_values"]
        # Defin time varying optimizer parameters
        self.parameters_nlp = param_dict["parameters_nlp"]
        # Initialize empty methods for completion later
        self.solver = []
        self.arg = []
        self.nlp_dict_out = []
        self.opt_result_step = []
        self.u_mpc = optimizer_model.ocp.u0
    @classmethod
    def user_optimizer(cls, optimizer_model, param_dict, *opt):
        "This method is open for the impelmentation of a user defined optimizer"
        dummy = 1
        return cls(dummy)
    def make_step(self):
        arg = self.arg
        result = self.solver(x0=arg['x0'], lbx=arg['lbx'], ubx=arg['ubx'], lbg=arg['lbg'], ubg=arg['ubg'], p = arg['p'])
        # Store the full solution
        self.opt_result_step = opt_result(result)
        # Extract the optimal control input to be applied
        nu = len(self.u_mpc)
        U_offset = self.nlp_dict_out['U_offset']
        v_opt = self.opt_result_step.optimal_solution
        self.u_mpc = NP.squeeze(v_opt[U_offset[0][0]:U_offset[0][0]+nu])
    def prepare_next(self, configuration):
        # TODO: do we really need here another function?
        observed_states = configuration.observer.observed_states
        X_offset = self.nlp_dict_out['X_offset']
        nx = len(configuration.model.ocp.x0)
        # Enforce the observed states as initial point for next optimization

        self.arg['lbx'][X_offset[0,0]:X_offset[0,0]+nx] = observed_states
        self.arg['ubx'][X_offset[0,0]:X_offset[0,0]+nx] = observed_states
        self.arg["x0"] = self.opt_result_step.optimal_solution
        # Pass as parameter the used control input
        self.arg['p'] = self.u_mpc
class observer:
    """A class for the definition model equations and optimal control problem formulation"""
    def __init__(self, model_observer, param_dict, *opt):
        self.x = param_dict['x']
        #self.observed_states = 0
    def make_step(self, configuration):
        self.make_measurement(configuration)
        self.observed_states = self.measurement # TODO: this is a dummy observer
    def make_measurement(self,configuration):
        # TODO: Here implement the own measurement function (or load it)
        # This is a dummy measurement
        self.measurement = configuration.simulator.xf_sim
    @classmethod
    def user_observer(cls, param_dict, *opt):
        " This is open for the implementation of a user-defined estimator class"
        dummy = 1
        return cls(dummy)

class configuration:
    """ A class for the definition of a do-mpc configuration that
    contains a model, optimizer, observer and simulator module """
    def __init__(self, model, optimizer, observer, simulator):
        self.model = model
        self.optimizer = optimizer
        self.observer = observer
        self.simulator = simulator

class mpc_data:
    "A class for the definition of the mpc data that is managed throughout the mpc loop"
    def __init__(self, configuration):
        # get sizes
        nx = configuration.model.x.size(1)
        nu = configuration.model.u.size(1)
        np = configuration.model.p.size(1)
        nz = configuration.model.p.size(1) # TODO: Adapt for DAEs
        t_end = configuration.optimizer.t_end
        t_step = configuration.simulator.t_step_simulator
        # Initialize the data structures
        self.mpc_states = NP.resize(NP.array([]),(t_end/t_step + 1,nx))
        self.mpc_control = NP.resize(NP.array([]),(t_end/t_step + 1,nu))
        self.mpc_alg = NP.resize(NP.array([]),(t_end/t_step + 1,nz))
        self.mpc_time = NP.resize(NP.array([]),(t_end/t_step + 1))
        self.mpc_cost = NP.resize(NP.array([]),(t_end/t_step + 1))
        self.mpc_ref = NP.resize(NP.array([]),(t_end/t_step + 1))
        self.mpc_cpu = NP.resize(NP.array([]),(t_end/t_step + 1))
        self.mpc_parameters = NP.resize(NP.array([]),(t_end/t_step + 1,np))
        # Initialize with initial conditions
        self.mpc_states[0,:] = configuration.model.ocp.x0
        self.mpc_control[0,:] = configuration.model.ocp.u0
        self.mpc_time[0] = 0
    def store_step(self, configuration):
        mpc_iter = configuration.simulator.mpc_iteration - 1 #Because already increased in the simulator
        self.mpc_states[mpc_iter,:] = configuration.simulator.xf_sim
        self.mpc_control[mpc_iter,:] = configuration.optimizer.u_mpc
        self.mpc_alg[mpc_iter,:] = 0 # TODO: To be completed for DAEs
        self.mpc_time[mpc_iter] = configuration.simulator.t0_sim # time already updated
        self.mpc_cost[mpc_iter] = configuration.optimizer.opt_result_step.optimal_cost
        self.mpc_ref[mpc_iter] = 0 # TODO: to be completed
        stats = configuration.optimizer.solver.stats()
        self.mpc_cpu[mpc_iter] = stats['t_wall_mainloop']
        self.mpc_parameters[mpc_iter,:] = configuration.simulator.p_real_now(configuration.simulator.t0_sim )

class opt_result:
    def __init__(self,res):
        self.optimal_solution = NP.array(res["x"])
        self.optimal_cost = NP.array(res["f"])
        self.constraints = NP.array(res["g"])


def setup_solver(configuration):

    # Call setup_nlp to generate the NLP
    nlp_dict_out = setup_nlp.setup_nlp(configuration.model, configuration.optimizer)
    # Set options
    opts = {}
    opts["expand"] = True
    opts["ipopt.linear_solver"] = configuration.optimizer.linear_solver
    #TODO: this should be passed as parameters of the optimizer class
    opts["ipopt.max_iter"] = 500
    opts["ipopt.tol"] = 1e-6
    # Setup the solver
    solver = nlpsol("solver", configuration.optimizer.nlp_solver, nlp_dict_out['nlp_fcn'], opts)
    arg = {}

    # Initial condition
    arg["x0"] = nlp_dict_out['vars_init']

    # Bounds on x
    arg["lbx"] = nlp_dict_out['vars_lb']
    arg["ubx"] = nlp_dict_out['vars_ub']

    # Bounds on g
    arg["lbg"] = nlp_dict_out['lbg']
    arg["ubg"] = nlp_dict_out['ubg']
    # NLP parameters
    arg["p"] = configuration.model.ocp.u0
    # TODO: better way than adding new fields here?
    configuration.optimizer.solver = solver
    configuration.optimizer.arg = arg
    configuration.optimizer.nlp_dict_out = nlp_dict_out
    return configuration
