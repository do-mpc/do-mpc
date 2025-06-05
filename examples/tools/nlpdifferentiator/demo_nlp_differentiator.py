
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

# %% Import required libraries
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
import os 
import sys

# Use dark background style for plots
# plt.style.use('dark_background')

# Add relative path to access do-mpc module
sys.path.append(os.path.join('..','..','..'))
import do_mpc

# %% Define the NLP problem for optimization
def get_optim():
    # Define decision variables (x) and parameter (p)
    x = ca.SX.sym('x', 2)
    p = ca.SX.sym('p', 1)

    # Define cost function (nonlinear Rosenbrock-type function)
    cost = (1 - x[0])**2 + 0.2 * (x[1] - x[0]**2)**2

    # Define nonlinear inequality constraints depending on x and p
    cons_inner = (x[0] + 0.5)**2 + x[1]**2
    cons = ca.vertcat(
        p**2 / 4 - cons_inner,
        cons_inner - p**2
    )

    # Bundle NLP components into a dict
    nlp = {'x': x, 'p': p, 'f': cost, 'g': cons}

    # Set bounds on variables and constraints
    nlp_bounds = {
        'lbx': np.array([0, -ca.inf]).reshape(-1, 1),      # Lower bounds on x
        'ubx': np.array([ca.inf, ca.inf]).reshape(-1, 1),  # Upper bounds on x
        'lbg': np.array([-ca.inf, -ca.inf]).reshape(-1, 1), # Lower bounds on constraints
        'ubg': np.array([0, 0]).reshape(-1, 1)              # Upper bounds on constraints
    }

    # Create an NLP solver using IPOPT
    solver = ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.print_level': 0,
        'ipopt.sb': 'yes',
        'print_time': 0,
        'ipopt.tol': 1e-14
    })

    return nlp, nlp_bounds, solver

# %% Setup NLP sensitivity analysis using do-mpc's NLPDifferentiator
def get_optim_diff(nlp, nlp_bounds):
    # Create NLPDifferentiator object
    nlp_diff = do_mpc.differentiator.NLPDifferentiator(nlp, nlp_bounds)

    # Enable constraint qualification checks and residual tracking
    nlp_diff.settings.check_LICQ = True
    nlp_diff.settings.check_rank = True
    nlp_diff.settings.track_residuals = True

    return nlp_diff

# Instantiate NLP and differentiator
nlp, nlp_bounds, solver = get_optim()
nlp_diff = get_optim_diff(nlp, nlp_bounds)

# %% Solve the NLP for different parameter values and compute sensitivities
p_test = np.linspace(0, 2.5, 50)            # Range of parameter values
x_test = np.zeros((len(p_test), 2))         # Store optimal x for each p
dxdp_test = np.zeros((len(p_test), 2))      # Store sensitivity dx/dp

# Loop over parameter values
for i, p_i in enumerate(p_test):
    print(i, end='\r')                      # Progress indication
    r = solver(p=p_i, **nlp_bounds)         # Solve NLP
    x_test[i] = r['x'].full().flatten()     # Store optimal x

    dxdp, _ = nlp_diff.differentiate(r, p_i)  # Compute dx/dp using differentiator
    dxdp_test[i] = dxdp.full().flatten()     # Store result

# %% Plot the optimal solutions and sensitivities
fig, ax = plt.subplots(2, sharex=True)      # Shared x-axis for subplot

colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']  # Get plot colors

# Plot optimal variables x0 and x1 as a function of p
ax[0].plot(p_test, x_test[:, 0], label='$x_0^*(p)$')
ax[1].plot(p_test, x_test[:, 1], label='$x_1^*(p)$')

# Plot sensitivities dx0/dp and dx1/dp
ax[0].plot(p_test, dxdp_test[:, 0], label='$\partial_p x_0^*(p)$')
ax[1].plot(p_test, dxdp_test[:, 1], label='$\partial_p x_1^*(p)$')

# Add quiver arrows to visualize the direction of sensitivities
every_nth = 5
s = slice(None, None, every_nth)
ax[0].quiver(p_test[s], x_test[s, 0], np.ones_like(p_test[s]), dxdp_test[s, 0], angles='xy', color=colors[1])
ax[1].quiver(p_test[s], x_test[s, 1], np.ones_like(p_test[s]), dxdp_test[s, 1], angles='xy', color=colors[1])

# Set labels and legends
ax[1].set_xlabel('$p$')
ax[0].legend()
ax[1].legend()
ax[0].set_title('Optimal solution and sensitivity depending on parameter $p$')

# Save figure
fig.savefig('demo_nlp_differentiator_dark.svg', format='svg')
plt.show(block=True)