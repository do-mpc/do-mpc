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

# imports
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import sys
sys.path.append('../../')
import do_mpc
import matplotlib.pyplot as plt
import logging
import cProfile
import pstats

# local imports
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

logging.basicConfig( level=logging.INFO)


# user settings
show_animation = False

# setting up the model
model = template_model()

# setting up a mpc controller, given the model
mpc = template_mpc(model)

# setting up a simulator, given the model
simulator = template_simulator(model)

# setting up an estimator, given the model
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
X_s_0 = 1.0 # This is the initial concentration inside the tank [mol/l]
S_s_0 = 0.5 # This is the controlled variable [mol/l]
P_s_0 = 0.0 #[C]
V_s_0 = 120.0 #[C]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])

# pushing initial condition to mpc, simulator and estimator
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# setting up initial guesses
mpc.set_initial_guess()

# run stats
LICQ_status_list = []
SC_status_list = []
residuals_list = []
param_sens_list = []
track_nlp_obj = []
track_nlp_res = []

# with cProfile.Profile() as pr:
pr = cProfile.Profile()
pr.enable()

# 
class ASMPC:
    """
    ASMPC: Approximate Sensitivity-based Model Predictive Controller
    
    This class implements a sensitivity-based approximation of the MPC solution
    by using a differentiator to compute the Jacobian of the optimal input with
    respect to the current state and previous control input. It leverages 
    CasADi's automatic differentiation through do-mpc's differentiator utility
    to compute an approximate optimal control input without solving the full
    nonlinear program at each time step.

    Parameters
    ----------
    mpc : do_mpc.controller.MPC
        A configured do-mpc MPC object.

    Attributes
    ----------
    mpc : do_mpc.controller.MPC
        The original MPC object used for control.
    nlp_diff : do_mpc.differentiator.DoMPCDifferentiator
        The differentiator object to compute sensitivities.
    _u_data : list of np.ndarray
        List storing the history of approximate control inputs.

    Methods
    -------
    make_step(x0)
        Computes the next control input using a sensitivity-based approximation
        instead of solving the full NLP.
    
    u_data
        Property returning the history of computed approximate control inputs.
    """
    def __init__(self, mpc):
        """
        Initialize the ASMPC class with a given do-mpc MPC object.

        Parameters
        ----------
        mpc : do_mpc.controller.MPC
            The do-mpc controller from which sensitivities and structure are derived.
        """
        self.mpc = mpc
        self.nlp_diff = do_mpc.differentiator.DoMPCDifferentiator(mpc)        
        self.nlp_diff.settings.check_LICQ = False
        self.nlp_diff.settings.check_rank = False
        self.nlp_diff.settings.lin_solver = 'scipy'

        self._u_data = [mpc.u0.cat.full().reshape(-1,1)]

    def make_step(self, x0):
        """
        Compute an approximate control input for the given state x0.

        This function uses first-order Taylor expansion with precomputed
        sensitivities to approximate the optimal input, thus avoiding full NLP
        solution. It updates internal history with the new input.

        Parameters
        ----------
        x0 : np.ndarray
            Current system state as a column vector.

        Returns
        -------
        u_next : np.ndarray
            Approximated optimal control input.
        """
        x0 = x0.reshape(-1,1)

        self.nlp_diff.differentiate()


        x_prev = self.mpc.x0.cat.full().reshape(-1,1)
        u0 = self.mpc.u0.cat.full().reshape(-1,1)
        u_prev = self.mpc.opt_p_num['_u_prev'].full().reshape(-1,1)

        

        du0dx0_num = self.nlp_diff.sens_num["dxdp", indexf["_u",0,0], indexf["_x0"]]
        du0du_prev_num = self.nlp_diff.sens_num["dxdp", indexf["_u",0,0], indexf["_u_prev"]].full()

        A = np.eye(self.mpc.model.n_u)-du0du_prev_num

        u_next  = np.linalg.inv(A)@(u0 + du0dx0_num @ (x0 - x_prev) - du0du_prev_num @ (u0))
        # u_next = u0 + du0dx0_num @ (x0 - x_prev) - du0du_prev_num @ (u0 - u_prev)

        self._u_data.append(u_next)

        return u_next
    
    @property
    def u_data(self):
        """
        Get the historical control inputs computed by the ASMPC approximation.

        Returns
        -------
        np.ndarray
            2D array of shape (n_u, n_steps), where each column is a control input.
        """
        return np.hstack(self._u_data)

# init of the asmpc class
asmpc = ASMPC(mpc)

# simulation of the plant
for k in range(30):
    
    if k>0:
        # for the current state x0, asmpc computes the approximate optimal control action u0
        u0_approx = asmpc.make_step(x0)

    # for the current state x0, mpc computes the optimal control action u0
    u0 = mpc.make_step(x0)

    # for the current state u0, computes the next state y_next
    y_next = simulator.make_step(u0)

    # for the current state y_next, estimates the next state x0
    x0 = estimator.make_step(y_next)

# stops profiling the above block and stores the stats
pr.disable()
stats = pstats.Stats(pr)

# configuring and showing plots
fig, ax = plt.subplots(1,1)
ax.plot(asmpc.u_data.T, '-x', label="approx")
ax.plot(mpc.data['_u'], '-x', label="mpc")
ax.legend()
plt.show(block=True)