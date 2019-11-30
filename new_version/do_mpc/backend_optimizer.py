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
from casadi import *
from casadi.tools import *
import pdb
import do_mpc.data


class backend_optimizer:
    def __init__(self):
        None

    def setup_discretization(self):
        _x, _u, _z, _tvp, _p, _aux = self.model.get_variables()
        # TODO: Now for testing
        # self.state_discretization = 'collocation'
        self.collocation_type = 'legendre'
        self.collocation_deg = 2
        self.collocation_ni = 2
        if self.state_discretization == 'discrete':
            ifcn = Function('ifcn', [_x, _u, _z, _tvp, _p], [[], self.model._rhs])
            n_total_coll_points = 0
        if self.state_discretization == 'collocation':
            ffcn = Function('ffcn', [_x, _u, _z, _tvp, _p], [self.model._rhs])
            # Get collocation information
            coll = self.collocation_type
            deg = self.collocation_deg
            ni = self.collocation_ni
            nk = self.n_horizon
            t_step = self.t_step
            n_x = self.model.n_x
            n_u = self.model.n_u
            n_p = self.model.n_p
            n_z = self.model.n_z
            n_tvp = self.model.n_tvp
            n_total_coll_points = (deg + 1) * ni
            # x_init = self._x0['x']

            # Choose collocation points
            if coll == 'legendre':    # Legendre collocation points
                tau_root = [0] + collocation_points(deg, 'legendre')
            elif coll == 'radau':     # Radau collocation points
                tau_root = [0] + collocation_points(deg, 'radau')
            else:
                raise Exception('Unknown collocation scheme')

            # Size of the finite elements
            h = t_step / ni

            # Coefficients of the collocation equation
            C = np.zeros((deg + 1, deg + 1))

            # Coefficients of the continuity equation
            D = np.zeros(deg + 1)

            # Dimensionless time inside one control interval
            tau = SX.sym("tau")

            # All collocation time points
            T = np.zeros((nk, ni, deg + 1))
            for k in range(nk):
                for i in range(ni):
                    for j in range(deg + 1):
                        T[k, i, j] = h * (k * ni + i + tau_root[j])

            # For all collocation points
            for j in range(deg + 1):
                # Construct Lagrange polynomials to get the polynomial basis at the
                # collocation point
                L = 1
                for r in range(deg + 1):
                    if r != j:
                        L *= (tau - tau_root[r]) / (tau_root[j] - tau_root[r])
                lfcn = Function('lfcn', [tau], [L])
                D[j] = lfcn(1.0)
                # Evaluate the time derivative of the polynomial at all collocation
                # points to get the coefficients of the continuity equation
                tfcn = Function('tfcn', [tau], [tangent(L, tau)])
                for r in range(deg + 1):
                    C[j, r] = tfcn(tau_root[r])

            # Initial condition
            xk0 = SX.sym("xk0", n_x)
            zk = SX.sym("zk", n_z)
            # Parameter
            pk = SX.sym("pk", n_p)
            tv_pk = SX.sym("tv_pk", n_tvp)
            # Control
            uk = SX.sym("uk", n_u)
            #uk_prev = SX.sym ("uk_prev",nu)
            # State trajectory
            n_ik = ni * (deg + 1) * n_x + n_x
            ik = SX.sym("ik", n_ik)
            ik_split = np.resize(np.array([], dtype=SX), (ni, deg + 1))

            # All variables with bounds and initial guess
            # ik_lb = np.zeros((n_ik,1))
            # ik_ub = np.zeros((n_ik,1))
            # ik_init = np.zeros((n_ik,1))
            offset = n_x

            # Store initial condition
            ik_split[0, 0] = ik[0 : n_x]
            first_j = 1  # Skip allocating x for the first collocation point for the first finite element

            # Penalty terms for the soft constraints
            # EPSILON = np.resize(np.array([], dtype=MX), (cons.size1()))

            # For each finite element
            for i in range(ni):
                # For each collocation point
                for j in range(first_j, deg + 1):
                    # Get the expression for the state vector
                    ik_split[i, j] = ik[offset:offset + n_x]

                    # Add the initial condition
                    # pdb.set_trace()
                    # ik_init[offset:offset + n_x] = x_init
                    #
                    # # Add bounds
                    # ik_lb[offset:offset + n_x] = x_lb
                    # ik_ub[offset:offset + n_x] = x_ub
                    offset += n_x

                # All collocation points in subsequent finite elements
                first_j = 0

            # Get the state at the end of the control interval
            xkf = ik[offset:offset + n_x]

            # # Add the initial condition
            # ik_init[offset:offset + n_x] = x_init
            #
            # # Add bounds
            # ik_lb[offset:offset + n_x] = x_lb
            # ik_ub[offset:offset + n_x] = x_ub
            offset += n_x

            # Check offset for consistency
            assert(offset == n_ik)

            # Constraints in the control interval
            gk = []
            lbgk = []
            ubgk = []

            # For all finite elements
            for i in range(ni):

                # For all collocation points
                for j in range(1, deg + 1):

                        # Get an expression for the state derivative at the
                        # collocation point
                    xp_ij = 0
                    for r in range(deg + 1):
                        xp_ij += C[r, j] * ik_split[i, r]

                    # Add collocation equations to the NLP
                    f_ij = ffcn(ik_split[i, j], uk, zk, tv_pk, pk)
                    gk.append(h * f_ij - xp_ij)
                    lbgk.append(np.zeros(n_x))  # equality constraints
                    ubgk.append(np.zeros(n_x))  # equality constraints

                # Get an expression for the state at the end of the finite element
                xf_i = 0
                for r in range(deg + 1):
                    xf_i += D[r] * ik_split[i, r]

                # Add continuity equation to NLP
                x_next = ik_split[i + 1, 0] if i + 1 < ni else xkf
                gk.append(x_next - xf_i)
                lbgk.append(np.zeros(n_x))
                ubgk.append(np.zeros(n_x))

            # Concatenate constraints
            gk = vertcat(*gk)
            lbgk = np.concatenate(lbgk)
            ubgk = np.concatenate(ubgk)

            assert(gk.shape[0] == ik.shape[0] - n_x) # Because now the initial point is part of ik

            # Create the integrator function
            ifcn = Function("ifcn", [ik, uk, zk, tv_pk, pk], [gk, xkf])

            # Return the integration function and the bounds for the collocaiton equations
        return ifcn, n_total_coll_points

    def setup_scenario_tree(self):
        """
        -----------------------------------------------------------------------------------
        Build the scenario tree given the possible values of the uncertain parmeters
        The strategy to build the tree is by default a combination of all the possible values
        This strategy can be modified by changing the code below
        -----------------------------------------------------------------------------------
        """
        n_p = self.model.n_p
        nk = self.n_horizon
        n_robust = self.n_robust
        # Build auxiliary variables that code the structure of the tree
        # Number of branches
        n_branches = [self.n_combinations if k < n_robust else 1 for k in range(nk)]
        # Calculate the number of scenarios (nodes at each stage)
        n_scenarios = [self.n_combinations**min(k, n_robust) for k in range(nk + 1)]
        # Scenaro tree structure
        child_scenario = -1 * np.ones((nk, n_scenarios[-1], n_branches [0])).astype(int)
        parent_scenario = -1 * np.ones((nk + 1, n_scenarios[-1])).astype(int)
        branch_offset = -1 * np.ones((nk, n_scenarios[-1])).astype(int)
        # Fill in the auxiliary structures
        for k in range(nk):
            # Scenario counter
            scenario_counter = 0
            # For all scenarios
            for s in range(n_scenarios[k]):
                # For all uncertainty realizations
                for b in range(n_branches[k]):
                    child_scenario[k][s][b] = scenario_counter
                    parent_scenario[k + 1][scenario_counter] = s
                    scenario_counter += 1
                # Store the range of branches
                if n_robust == 0:
                    branch_offset[k][s] = 0
                elif k < n_robust:
                    branch_offset[k][s] = 0
                else:
                    branch_offset[k][s] = s % n_branches[0]

        return n_branches, n_scenarios, child_scenario, parent_scenario, branch_offset

    def setup_nlp(self):
        self.check_validity()
        # Obtain an integrator (collocation, discrete-time) and the amount of intermediate (collocation) points
        ifcn, n_total_coll_points = self.setup_discretization()
        n_branches, n_scenarios, child_scenario, parent_scenario, branch_offset = self.setup_scenario_tree()
        n_max_scenarios = self.n_combinations ** self.n_robust
        # Create struct for optimization variables:
        self.opt_x = opt_x = struct_symSX([
            entry('_x', repeat=[self.n_horizon+1, n_max_scenarios, 1+n_total_coll_points], struct=self.model._x),
            entry('_z', repeat=[self.n_horizon, n_max_scenarios, 1+n_total_coll_points], struct=self.model._z),
            entry('_u', repeat=[self.n_horizon, n_max_scenarios], struct=self.model._u),
        ])
        # FIXME: Currently there exist dummy collocation points for the terminal state

        # Create struct for optimization parameters:
        self.opt_p = opt_p = struct_symSX([
            entry('_x0', struct=self.model._x),
            entry('_tvp', repeat=self.n_horizon, struct=self.model._tvp),
            entry('_p', repeat=self.n_combinations, struct=self.model._p),
            entry('_u_prev', struct=self.model._u)
        ])

        self.lb_opt_x = opt_x(-np.inf)
        self.ub_opt_x = opt_x(np.inf)

        # Initialize objective function and constraints
        obj = 0
        cons = []
        cons_lb = []
        cons_ub = []

        # Initial condition:
        cons.append(opt_x['_x', 0, 0, 0]-opt_p['_x0'])

        cons_lb.append(np.zeros((self.model.n_x, 1)))
        cons_ub.append(np.zeros((self.model.n_x, 1)))

        # TODO: Weigthing factors for the tree assumed equal. They could be set from outside
        # Weighting factor for every scenario
        omega = [1. / n_scenarios[k + 1] for k in range(self.n_horizon)]
        omega_delta_u = [1. / n_scenarios[k + 1] for k in range(self.n_horizon)]

        # For all control intervals
        for k in range(self.n_horizon):
            # For all scenarios (grows exponentially with n_robust)
            for s in range(n_scenarios[k]):
                # For all childen nodes of each node at stage k, discretize the model equations
                for b in range(n_branches[k]):
                    # Obtain the index of the parameter values that should be used for this scenario
                    current_scenario = b + branch_offset[k][s]

                    # Add constraints for state equation:
                    [g_ksb, xf_ksb] = ifcn(vertcat(*opt_x['_x', k, s, :]), opt_x['_u', k, s], vertcat(*opt_x['_z', k, s, :]), opt_p['_tvp', k], opt_p['_p', current_scenario])

                    # Add the collocation equations
                    cons.append(g_ksb)
                    cons_lb.append(np.zeros(g_ksb.shape[0]))
                    cons_ub.append(np.zeros(g_ksb.shape[0]))

                    # Add continuity constraints
                    cons.append(xf_ksb - opt_x['_x', k+1, child_scenario[k][s][b], 0])
                    cons_lb.append(np.zeros((self.model.n_x, 1)))
                    cons_ub.append(np.zeros((self.model.n_x, 1)))

                    # Add nonlinear constraints only on each control step
                    nl_cons_k = self._nl_cons_fun(opt_x['_x', k, s, 0], opt_x['_u', k, s], opt_x['_z', k, s, 0], opt_p['_tvp', k], opt_p['_p', current_scenario])
                    cons.append(nl_cons_k)
                    cons_lb.append(self._nl_cons_lb)
                    cons_ub.append(self._nl_cons_ub)

                    # Add terminal constraints
                    # TODO: Add terminal constraints with an additional nl_cons

                    # Add contribution to the cost
                    obj += omega[k] * self.lterm_fun(opt_x['_x', k, s, 0], opt_x['_u', k, s], opt_x['_z', k, s, 0], opt_p['_tvp', k], opt_p['_p', current_scenario])
                    # In the last step add the terminal cost too
                    if k == self.n_horizon - 1:
                        obj += omega[k] * self.mterm_fun(opt_x['_x', k + 1, s, 0])

                    # U regularization:
                    if k == 0:
                        # pdb.set_trace()
                        obj += self.rterm_factor.cat.T@((opt_x['_u', 0, s]-opt_p['_u_prev'])**2)
                    else:
                        obj += self.rterm_factor.cat.T@((opt_x['_u', k, s]-opt_x['_u', k-1, parent_scenario[k][s]])**2)

                # Bounds for the states on all discretize values along the horizon
                self.lb_opt_x['_x', k, s, :] = self._x_lb
                self.ub_opt_x['_x', k, s, :] = self._x_ub

                # Bounds for the inputs along the horizon
                self.lb_opt_x['_u', k, s] = self._u_lb
                self.ub_opt_x['_u', k, s] = self._u_ub

                # Bounds on the terminal state
                if k == self.n_robust - 1:
                    self.lb_opt_x['_x', self.n_horizon, s, 0] = self._x_lb
                    self.ub_opt_x['_x', self.n_horizon, s, 0] = self._x_ub

        cons = vertcat(*cons)
        self.cons_lb = vertcat(*cons_lb)
        self.cons_ub = vertcat(*cons_ub)

        # Create casadi optimization object:
        optim_opts = {}
        optim_opts["expand"] = False
        optim_opts["ipopt.linear_solver"] = 'ma27'
        nlp = {'x': vertcat(opt_x), 'f': obj, 'g': cons, 'p': vertcat(opt_p)}
        self.S = nlpsol('S', 'ipopt', nlp, optim_opts)

        # Create copies of these structures with numerical values (all zero):
        self.opt_x_num = self.opt_x(0)
        self.opt_p_num = self.opt_p(0)
