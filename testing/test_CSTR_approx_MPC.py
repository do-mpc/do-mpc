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
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import unittest
import pickle
import torch

from importlib import reload
import copy

do_mpc_path = '../'
if not do_mpc_path in sys.path:
    sys.path.append('../')

import do_mpc

from do_mpc.approximateMPC import AMPCSampler
from do_mpc.approximateMPC import ApproxMPC, Trainer



class TestapproxCSTR(unittest.TestCase):
    def setUp(self):
        """Add path of test case and import the modules.
        If this test isn't the first to run, the modules need to be reloaded.
        Reset path afterwards.
        """

        default_path = copy.deepcopy(sys.path)
        sys.path.append('../examples/CSTR_approximate_mpc/')
        import template_mpc
        import template_model
        import template_simulator

        self.template_model = reload(template_model)
        self.template_mpc = reload(template_mpc)
        self.template_simulator = reload(template_simulator)
        sys.path = default_path

    def test_SX(self):
        print('Testing SX implementation')
        model = self.template_model.template_model('SX')
        self.approxCSTR(model)

    def test_MX(self):
        print('Testing MX implementation')
        model = self.template_model.template_model('MX')
        self.approxCSTR(model)

    def test_pickle_unpickle(self):
        print('Testing SX implementation with pickle / unpickle')
        # Test if pickling / unpickling works for the SX model:
        model = self.template_model.template_model('SX')
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Load the casadi structure
        with open('model.pkl', 'rb') as f:
            model_unpickled = pickle.load(f)
        self.approxCSTR(model_unpickled)

    def approxCSTR(self, model):
        """
        Get configured do-mpc modules:
        For now we only test if everything runs through without errors.
        """

        # setting up a mpc controller, given the model
  

        # setting up a mpc controller, given the model
        mpc = self.template_mpc.template_mpc(model, silence_solver=True)

        # setting up a simulator, given the model
        simulator = self.template_simulator.template_simulator(model)

        # setting up an estimator, given the model
        estimator = do_mpc.estimator.StateFeedback(model)

        # Set the initial state of mpc and simulator:
        C_a_0 = 0.8  # This is the initial concentration inside the tank [mol/l]
        C_b_0 = 0.5  # This is the controlled variable [mol/l]
        T_R_0 = 134.14  # [C]
        T_K_0 = 130.0  # [C]
        x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1, 1)
        u0 = np.array([5.0, 0.0]).reshape(-1, 1)

        # pushing initial condition to mpc and the simulator
        mpc.u0=u0
        mpc.x0 = x0
        simulator.x0 = x0

        # setting up initial guesses
        mpc.set_initial_guess()
        simulator.set_initial_guess() 

        # approximate mpc initialization
        approx_mpc = ApproxMPC(mpc)

        # configuring approximate mpc settings
        approx_mpc.settings.n_hidden_layers = 1
        approx_mpc.settings.n_neurons = 50

        # approximate mpc setup
        approx_mpc.setup()


        # initializing sampler for the approximate mpc
        sampler = AMPCSampler(mpc)

        dataset_name = 'my_dataset_new'
        # configuring sampler settings
        n_samples = 50
        sampler.settings.closed_loop_flag = True
        sampler.settings.trajectory_length = 1
        sampler.settings.n_samples = n_samples
        sampler.settings.dataset_name = dataset_name

        # sampler setup
        sampler.setup()

        # generating the samples
        np.random.seed(42)  # for reproducibility
        sampler.default_sampling()

        # initializing trainer for the approximate mpc
        trainer = Trainer(approx_mpc)

        # configuring trainer settings
        trainer.settings.dataset_name = dataset_name
        trainer.settings.n_epochs = 10
        trainer.settings.show_fig =False
        trainer.settings.save_fig = False
        trainer.settings.save_history = False

        # configuring scheduler settings
        trainer.settings.scheduler_flag = True
        trainer.scheduler_settings.cooldown = 0
        trainer.scheduler_settings.patience = 50

        # trainer setup
        trainer.setup()


        # training the approximate mpc with the sampled data
        torch.manual_seed(42)  # for reproducibility
        trainer.default_training()

        # pushing initial condition to approx_mpc
        approx_mpc.u0=u0

        # simulation of the plant
        sim_time=10
        for k in range(sim_time):

            # for the current state x0, approx_mpc computes the optimal control action u0
            u0 = approx_mpc.make_step(x0,clip_to_bounds=True)

            # for the current state u0, computes the next state y_next
            y_next = simulator.make_step(u0)

            # for the current state y_next, computes the next state x0
            x0 = estimator.make_step(y_next)



        return




if __name__ == '__main__':
    unittest.main()
