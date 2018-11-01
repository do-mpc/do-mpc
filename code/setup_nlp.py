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
#
#   Important parts of this script were coded in colaboration with Joel Andersson.
#   his support is gratefully acknowledged

from casadi import *
from casadi.tools import *
import numpy as NP
import core_do_mpc
from copy import deepcopy
import pdb


def setup_nlp(model, optimizer):

    # Decode all the necessary parameters from the model and optimizer information
    # NOTE: The names of some variables are not consistent. But not critical
    # Parameters from optimizer
    nk = optimizer.n_horizon
    n_robust = optimizer.n_robust
    t_step = optimizer.t_step
    deg = optimizer.poly_degree
    coll = optimizer.collocation
    ni = optimizer.n_fin_elem
    open_loop = optimizer.open_loop
    uncertainty_values = optimizer.uncertainty_values
    #parameters_nlp = optimizer.parameters_nlp
    state_discretization = optimizer.state_discretization
    # Parameters from model
    x0 = model.ocp.x0
    u0 = model.ocp.u0
    x_lb = model.ocp.x_lb
    x_ub = model.ocp.x_ub
    u_lb = model.ocp.u_lb
    u_ub = model.ocp.u_ub
    x_scaling = model.ocp.x_scaling
    u_scaling = model.ocp.u_scaling
    cons = model.ocp.cons
    cons_ub = model.ocp.cons_ub
    cons_terminal = model.ocp.cons_terminal
    cons_terminal_lb = model.ocp.cons_terminal_lb
    cons_terminal_ub = model.ocp.cons_terminal_ub
    soft_constraint = model.ocp.soft_constraint
    penalty_term_cons = model.ocp.penalty_term_cons
    maximum_violation = model.ocp.maximum_violation
    mterm = model.ocp.mterm
    lterm = model.ocp.lterm
    rterm = model.ocp.rterm
    x = model.x
    u = model.u
    p = model.p
    z = model.z
    tv_p = model.tv_p
    xdot = model.rhs

    # Size of the state, control and parameter vector

    nx = x.size(1)
    nu = u.size(1)
    np = p.size(1)
    ntv_p = tv_p.size(1)

    # Generate, scale and initialize all the necessary functions
    # Consider as initial guess the initial conditions
    x_init = deepcopy(x0)
    u_init = deepcopy(u0)
    up = vertcat(u, p)

    # Right hand side of the ODEs
    # NOTE: look scaling (appears to be fine)
    for i in (x0, x_ub, x_lb, x_init):
        i /= x_scaling
    xdot = substitute(xdot, x, x * x_scaling) / x_scaling
    for i in (u_ub, u_lb, u_init):
        i /= u_scaling
    xdot = substitute(xdot, u, u * u_scaling)
    ffcn = Function('ffcn', [x, up, tv_p], [xdot])

    # Constraints, possibly soft
    # Epsilon for the soft constraints
    cons = substitute(cons, x, x * x_scaling)
    cons = substitute(cons, u, u * u_scaling)
    if soft_constraint:
        epsilon = SX.sym("epsilon", cons.size1())
        cons = cons - epsilon
        cfcn = Function('cfcn', [x, u, p, epsilon, tv_p], [cons])
    else:
        cfcn = Function('cfcn', [x, u, p, tv_p], [cons])
    cons_terminal = substitute(cons_terminal, x, x * x_scaling)
    cons_terminal = substitute(cons_terminal, u, u * u_scaling)
    cfcn_terminal = Function('cfcn', [x, u, p], [cons_terminal])
    # Mayer term of the cost functions
    mterm = substitute(mterm, x, x * x_scaling)
    mterm = substitute(mterm, u, u * u_scaling)
    mfcn = Function('mfcn', [x, u, p, tv_p], [mterm])
    # Lagrange term of the cost function
    lterm = substitute(lterm, x, x * x_scaling)
    lterm = substitute(lterm, u, u * u_scaling)
    lagrange_fcn = Function('lagrange_fcn', [x, u, p, tv_p], [lterm])
    # Penalty term for the control inputs
    u_prev = SX.sym("u_prev", nu)
    du = u - u_prev
    R = diag(SX(rterm))
    rterm = substitute(rterm, x, x * x_scaling)
    rterm = substitute(rterm, u, u * u_scaling)
    rfcn = Function('rfcn', [u_prev, u], [mtimes(du.T, mtimes(R, du))])
    """
    -----------------------------------------------------------------------------------
    Build the scenario tree given the possible values of the uncertain parmeters
    The strategy to build the tree is by default a combination of all the possible values
    This strategy can be modified by changing the code below
    -----------------------------------------------------------------------------------
    """

    # Initialize some auxiliary variables
    current_scenario = NP.resize(NP.array([], dtype=int), np)
    p_scenario_index = NP.resize(NP.array([], dtype=int), np)
    p_scenario = NP.resize(NP.array([]), np)
    number_values_per_uncertainty = NP.resize(NP.array([], dtype=int), np)
    k = 1
    # Get the number of different values of each parameter
    for ii in range(np):
        number_values_per_uncertainty[ii] = uncertainty_values[ii].size
        current_scenario[ii] = 0
    while (current_scenario != number_values_per_uncertainty - 1).any():
        for index in range(np - 1, -1, -1):
            if current_scenario[index] + \
                    1 < number_values_per_uncertainty[index]:
                # If it is no the last element increase it and break the for
                # loop
                current_scenario[index] += NP.array([1])
                break
            else:
                current_scenario[index] = NP.array([0])
        # Add the current scenario to the variable p_scenario
        p_scenario_index = NP.vstack((p_scenario_index, current_scenario))

    # Initialize the vector with the real values
    p_scenario = deepcopy(p_scenario_index) * 1.0
    for jj in range(len(p_scenario_index)):
        for ii in range(np):
            p_scenario[jj, ii] = uncertainty_values[ii][p_scenario_index[jj, ii]]

    """
    -----------------------------------------------------------------------------------
    End of the strategy to generate the scenario tree. The different scenarios are
    stored in the variable p_scenario
    -----------------------------------------------------------------------------------
    """

    # Collocation discretization
    if state_discretization == 'collocation':

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
        C = NP.zeros((deg + 1, deg + 1))

        # Coefficients of the continuity equation
        D = NP.zeros(deg + 1)

        # Dimensionless time inside one control interval
        tau = SX.sym("tau")

        # All collocation time points
        T = NP.zeros((nk, ni, deg + 1))
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
        xk0 = MX.sym("xk0", nx)

        # Parameter
        pk = MX.sym("pk", np)
        tv_pk = MX.sym("tv_pk", ntv_p)
        # Control
        uk = MX.sym("uk", nu)
        #uk_prev = MX.sym ("uk_prev",nu)
        # State trajectory
        n_ik = ni * (deg + 1) * nx
        ik = MX.sym("ik", n_ik)
        ik_split = NP.resize(NP.array([], dtype=MX), (ni, deg + 1))

        # All variables with bounds and initial guess
        ik_lb = NP.zeros(n_ik)
        ik_ub = NP.zeros(n_ik)
        ik_init = NP.zeros(n_ik)
        offset = 0

        # Store initial condition
        ik_split[0, 0] = xk0
        first_j = 1  # Skip allocating x for the first collocation point for the first finite element

        # Penalty terms for the soft constraints
        EPSILON = NP.resize(NP.array([], dtype=MX), (cons.size1()))

        # For each finite element
        for i in range(ni):
            # For each collocation point
            for j in range(first_j, deg + 1):
                # Get the expression for the state vector
                ik_split[i, j] = ik[offset:offset + nx]

                # Add the initial condition
                ik_init[offset:offset + nx] = x_init

                # Add bounds
                ik_lb[offset:offset + nx] = x_lb
                ik_ub[offset:offset + nx] = x_ub
                offset += nx

            # All collocation points in subsequent finite elements
            first_j = 0

        # Get the state at the end of the control interval
        xkf = ik[offset:offset + nx]

        # Add the initial condition
        ik_init[offset:offset + nx] = x_init

        # Add bounds
        ik_lb[offset:offset + nx] = x_lb
        ik_ub[offset:offset + nx] = x_ub
        offset += nx

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
                [f_ij] = ffcn.call([ik_split[i, j], vertcat(uk, pk), tv_pk])
                gk.append(h * f_ij - xp_ij)
                lbgk.append(NP.zeros(nx))  # equality constraints
                ubgk.append(NP.zeros(nx))  # equality constraints

            # Get an expression for the state at the end of the finite element
            xf_i = 0
            for r in range(deg + 1):
                xf_i += D[r] * ik_split[i, r]

            # Add continuity equation to NLP
            x_next = ik_split[i + 1, 0] if i + 1 < ni else xkf
            gk.append(x_next - xf_i)
            lbgk.append(NP.zeros(nx))
            ubgk.append(NP.zeros(nx))

        # Concatenate constraints
        gk = vertcat(*gk)
        lbgk = NP.concatenate(lbgk)
        ubgk = NP.concatenate(ubgk)
        assert(gk.size() == ik.size())

        # Create the integrator function
        ifcn = Function("ifcn", [ik, xk0, pk, uk, tv_pk], [gk, xkf])
    # FIXME: update so that multiple_shooting works
    elif state_discretization == 'multiple-shooting':

        # Create an integrator instance
        #   ifcn = Integrator('cvodes',ffcn)
        #   ifcn.setOption("exact_jacobian",True)
        #   ifcn.setOption("reltol",1e-8)
        #   ifcn.setOption("abstol",1e-8)

        # Set options
        # ifcn.setOption("tf",t_step)
        x_ms = MX.sym('x_ms', nx)
        u_ms = MX.sym('u_ms', nu)
        p_ms = MX.sym('p_ms', np)
        xdot_ms = x_ms
        dae = {'x': x_ms, 'p': vertcat(u_ms, p_ms), 'ode': xdot_ms}
        opts = {
            "abstol": 1e-10,
            "reltol": 1e-10,
            "exact_jacobian": True,
            'tf': t_step}
        ifcn = integrator("simulator_ms", 'cvodes', dae, opts)

        # No implicitly defined variables
        n_ik = 0
        # Penalty terms for the soft constraints
        EPSILON = NP.resize(NP.array([], dtype=MX), (cons.size1()))
        #uk_prev = MX.sym ("uk_prev",nu)

    elif state_discretization == 'discrete-time':
        # no need to define an integrator for the discrete-time case
        # No implicitly defined variables
        n_ik = 0
        # Penalty terms for the soft constraints
        EPSILON = NP.resize(NP.array([], dtype=MX), (cons.size1()))
        #uk_prev = MX.sym ("uk_prev",nu)
        pass

    # Number of branches
    n_branches = [len(p_scenario) if k < n_robust else 1 for k in range(nk)]

    # Calculate the number of scenarios for x and u
    n_scenarios = [len(p_scenario)**min(k, n_robust) for k in range(nk + 1)]
    # Scenaro tree structure
    child_scenario = NP.resize(
        NP.array([-1], dtype=int), (nk, n_scenarios[-1], n_branches[0]))
    parent_scenario = NP.resize(
        NP.array([-1], dtype=int), (nk + 1, n_scenarios[-1]))
    branch_offset = NP.resize(NP.array([-1], dtype=int), (nk, n_scenarios[-1]))
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

    # Count the total number of variables
    NV = len(p_scenario) * np
    for k in range(nk):
        NV += n_scenarios[k] * (nu + nx + n_branches[k] * n_ik)
    NV += n_scenarios[nk] * nx  # End point

    if soft_constraint:
                    # If soft constraints are implemented
        NV += cons.size1()
    # Weighting factor for every scenario
    omega = [1. / n_scenarios[k + 1] for k in range(nk)]
    omega_delta_u = [1. / n_scenarios[k + 1] for k in range(nk)]
    #omega_delta_u[0] =1./n_scenarios[0+1]

    # NLP variable vector
    V = MX.sym("V", NV)

    # All variables with bounds and initial guess
    vars_lb = NP.zeros(NV)
    vars_ub = NP.zeros(NV)
    vars_init = NP.zeros(NV)
    offset = 0

    # Get parameters
    P = NP.resize(NP.array([], dtype=MX), (len(p_scenario)))
    for b in range(len(p_scenario)):
        P[b] = V[offset:offset + np]
        vars_lb[offset:offset + np] = p_scenario[b]
        vars_ub[offset:offset + np] = p_scenario[b]
        offset += np

    # Get collocated states and parametrized control
    X = NP.resize(NP.array([], dtype=MX), (nk + 1, n_scenarios[-1]))
    if state_discretization == 'collocation':
        I = NP.resize(NP.array([], dtype=MX),
                      (nk, n_scenarios[-1], n_branches[0]))
    U = NP.resize(NP.array([], dtype=MX), (nk, n_scenarios[-1]))
    parameters_setup_nlp = struct_symMX(
        [entry("uk_prev", shape=(nu)), entry("TV_P", shape=(ntv_p, nk))])
    TV_P = parameters_setup_nlp['TV_P']
    uk_prev = parameters_setup_nlp['uk_prev']
    # The offset variables contain the position of the states and controls in
    # the vector of opt. variables
    X_offset = NP.resize(NP.array([-1], dtype=int), X.shape)
    U_offset = NP.resize(NP.array([-1], dtype=int), U.shape)
    E_offset = NP.resize(NP.array([-1], dtype=int), EPSILON.shape)
    for k in range(nk):
        # For all scenarios
        for s in range(n_scenarios[k]):
            # Get the expression for the state vector
            X[k, s] = V[offset:offset + nx]
            X_offset[k, s] = offset

            # Add the initial condition
            vars_init[offset:offset + nx] = x_init

            if k == 0:
                vars_lb[offset:offset + nx] = x0
                vars_ub[offset:offset + nx] = x0

            else:
                vars_lb[offset:offset + nx] = x_lb
                vars_ub[offset:offset + nx] = x_ub
            offset += nx

            # State trajectory if collocation
            if state_discretization == 'collocation':

                # For all uncertainty realizations
                for b in range(n_branches[k]):
                    # Get an expression for the implicitly defined variables
                    I[k, s, b] = V[offset:offset + n_ik]

                    # Add the initial condition and bounds
                    vars_init[offset:offset + n_ik] = ik_init
                    vars_lb[offset:offset + n_ik] = ik_lb
                    vars_ub[offset:offset + n_ik] = ik_ub
                    offset += n_ik

            # Parametrized controls
            U[k, s] = V[offset:offset + nu]
            U_offset[k, s] = offset
            vars_lb[offset:offset + nu] = u_lb
            vars_ub[offset:offset + nu] = u_ub
            vars_init[offset:offset + nu] = u_init
            offset += nu

    # State at end time (for all x scenarios) This can be modified in case
    # they are different
    for s in range(n_scenarios[nk]):
        X[nk, s] = V[offset:offset + nx]
        X_offset[nk, s] = offset
        vars_lb[offset:offset + nx] = x_lb
        vars_ub[offset:offset + nx] = x_ub
        vars_init[offset:offset + nx] = x_init
        offset += nx
    if soft_constraint:
            # Last elements (epsilon) for soft constraints
        EPSILON = V[offset:offset + cons.size1()]
        E_offset = offset
        vars_lb[offset:offset + cons.size1()] = NP.zeros(cons.size1())
        vars_ub[offset:offset + cons.size1()] = maximum_violation
        vars_init[offset:offset + cons.size1()] = 0
        offset += cons.size1()

    # Check offset for consistency
    assert(offset == NV)

    # Constraint function for the NLP
    g = []
    lbg = []
    ubg = []

    # Objective function in the NLP
    J = 0

    # For all control intervals
    for k in range(nk):
        # For all scenarios
        for s in range(n_scenarios[k]):

            # Initial state and control
            X_ks = X[k, s]
            U_ks = U[k, s]

            # For all uncertainty realizations
            for b in range(n_branches[k]):

                # Parameter realization
                P_ksb = P[b + branch_offset[k][s]]

                if state_discretization == 'collocation':

                    # Call the inlined integrator
                    [g_ksb, xf_ksb] = ifcn.call(
                        [I[k, s, b], X_ks, P_ksb, U_ks, TV_P[:, k]])

                    # Add equations defining the implicitly defined variables
                    # (i.e. collocation and continuity equations) to the NLP
                    g.append(g_ksb)
                    lbg.append(NP.zeros(n_ik))  # equality constraints
                    ubg.append(NP.zeros(n_ik))  # equality constraints

                elif state_discretization == 'multiple-shooting':

                    # Call the integrator
                    # FIXME: update so that multiple-shooting works
                    # pdb.set_trace()
                    ifcn_out = ifcn(x0=X_ks, p=vertcat(U_ks, P_ksb))
                    xf_ksb = ifcn_out['xf']

                elif state_discretization == 'discrete-time':
                    [xf_ksb] = ffcn.call(
                        [X_ks, vertcat(U_ks, P_ksb), TV_P[:, k]])

                # Add continuity equation to NLP
                g.append(X[k + 1, child_scenario[k][s][b]] - xf_ksb)
                lbg.append(NP.zeros(nx))
                ubg.append(NP.zeros(nx))

                # Add extra constraints depending on other states
                # pdb.set_trace()
                if soft_constraint:
                    [residual] = cfcn.call(
                        [xf_ksb, U_ks, P_ksb, EPSILON, TV_P[:, k]])
                else:
                    [residual] = cfcn.call(
                        [xf_ksb, U_ks, P_ksb, TV_P[:, k]])
                g.append(residual)
                lbg.append(NP.ones(cons.size1()) * (-inf))
                ubg.append(cons_ub)

                # Add terminal constraints
                if k == nk - 1:
                    [residual_terminal] = cfcn_terminal.call(
                        [xf_ksb, U_ks, P_ksb])
                    g.append(residual_terminal)
                    lbg.append(cons_terminal_lb)
                    ubg.append(cons_terminal_ub)
                # Add contribution to the cost
                if k < nk - 1:
                    [J_ksb] = lagrange_fcn.call(
                        [xf_ksb, U_ks, P_ksb, TV_P[:, k]])
                else:
                    [J_ksb] = mfcn.call([xf_ksb, U_ks, P_ksb, TV_P[:, k]])
                J += omega[k] * J_ksb

                # Add contribution to the cost of the soft constraints penalty
                # term
                if soft_constraint:
                        # pdb.set_trace()
                    for index_soft in range(cons.size1()):
                        J_ksb_soft = penalty_term_cons[index_soft] * \
                            (EPSILON[index_soft])**2
                        J += J_ksb_soft
                # Penalize deviations in u
                s_parent = parent_scenario[k][s]
                u_prev = U[k - 1, s_parent] if k > 0 else uk_prev
                [du_k] = rfcn.call([u_prev, U[k, s]])
                J += omega_delta_u[k] * n_branches[k] * du_k

    # Add non-anticipativity constraints for open-loop multi-stage NMPC
    if open_loop == 1:
        for kk in range(1, nk):
            for ss in range(n_scenarios[kk] - 1):
                g.append(U[kk, ss] - U[kk, ss + 1])
                lbg.append(NP.zeros(nu))
                ubg.append(NP.zeros(nu))
    # Concatenate constraints
    g = vertcat(*g)
    # pdb.set_trace()
    lbg = vertcat(*lbg)
    ubg = vertcat(*ubg)

    nlp_fcn = {'f': J, 'x': V, 'p': parameters_setup_nlp, 'g': g}

    nlp_dict_out = {
        'nlp_fcn': nlp_fcn,
        'X_offset': X_offset,
        'U_offset': U_offset,
        'E_offset': E_offset,
        'vars_lb': vars_lb,
        'vars_ub': vars_ub,
        'vars_init': vars_init,
        'lbg': lbg,
        'ubg': ubg,
        'parent_scenario': parent_scenario,
        'child_scenario': child_scenario,
        'n_branches': n_branches,
        'n_scenarios': n_scenarios,
        'p_scenario': p_scenario}

    return nlp_dict_out
