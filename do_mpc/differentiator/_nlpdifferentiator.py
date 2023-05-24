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
import scipy.linalg as sp_linalg
import scipy.sparse as sp_sparse
import casadi as ca
import casadi.tools as castools 
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional, Any
import pdb

from do_mpc.optimizer import Optimizer
from .helper import NLPDifferentiatorSettings, NLPDifferentiatorStatus

import logging

__all__ = ['NLPDifferentiator', 'DoMPCDifferentiator']

class NLPDifferentiator:
    """
    Base class for nonlinear program (NLP) differentiator. 
    This class can be used independently from ``do-mpc`` to differentiate a given NLP.
 
    Note:
        This is an experimental feature. The API might change in the future. 


    **Example**:

    1. Consider an NLP created with CasADi, including
        - optimization variables ``x``
        - optimization parameters ``p``
        - objective function ``f``
        - constraints ``g``

    :: 

        import casadi as ca

        x = ca.SX.sym('x', 2)
        p = ca.SX.sym('p', 1)
        f = (1-x[0])**2 + 0.2*(x[1]-x[0]**2)**2
        cons_inner = (x[0] + 0.5)**2+x[1]**2

        g = ca.vertcat(
            p**2/4 - cons_inner,
            cons_inner - p**2
        )

        ca_solver = ca.nlpsol('solver', 'ipopt', nlp)

    2. Create dictionaries for the NLP and the NLP bounds:

    ::
    
        nlp = {'x':x, 'p':p, 'f':cost, 'g':cons}
        nlp_bounds = {
            'lbx': np.array([0, -ca.inf]).reshape(-1,1), 
            'ubx': np.array([ca.inf, ca.inf]).reshape(-1,1), 
            'lbg': np.array([-ca.inf, -ca.inf]).reshape(-1,1), 
            'ubg': np.array([0, 0]).reshape(-1,1)
        }

    3. Initialize the NLP differentiator with the NLP and the NLP bounds. 

    ::

        nlp_diff = NLPDifferentiator(nlp, nlp_bounds)

    4. Configure the differentiator settings with the :py:attr:`settings` attribute.

    ::

        nlp_diff.settings.check_LICQ = False

    5. Solve the parametric NLP for the parameters ``p0``, e.g. with the CasADi solver ``ca_solver``. Pass the same bounds that were used previously.

    ::

        p0 = np.array([1.])
        r = solver(p=p0, **nlp_bounds)

    6. Calculate the parametric NLP sensitivity matrices with :py:meth:`differentiate` considering the solution ``r`` and the corresponding parameters ``p0``.

    ::

        dxdp, dlamdp = nlp_diff.get_sensitivity_matrices(r, p0)


    Args:
        nlp : Dictionary with keys ``x``, ``p``, ``f``, ``g``.
        nlp_bounds : Dictionary with keys ``lbx``, ``ubx``, ``lbg``, ``ubg``.

    """
    def __init__(self, nlp: Dict, nlp_bounds: Dict, **kwargs):

        nlp_mandatory_keys = ['f', 'x', 'p', 'g']
        nlp_bounds_mandatory_keys = ['lbx', 'ubx', 'lbg', 'ubg']
        
        if not isinstance(nlp, dict):
            raise ValueError('nlp must be a dictionary.')

        if not isinstance(nlp_bounds, dict):
            raise ValueError('nlp_bounds must be a dictionary.')

        if not set(nlp.keys()).issuperset(set(nlp_mandatory_keys)):
            raise ValueError('nlp must contain keys {}.'.format(nlp_mandatory_keys))

        if not set(nlp_bounds.keys()).issuperset(set(nlp_bounds_mandatory_keys)):
            raise ValueError('nlp_bounds must contain keys {}.'.format(nlp_bounds_mandatory_keys))
        
        self.nlp = nlp.copy()
        self.nlp_bounds = nlp_bounds.copy()

        self._status = NLPDifferentiatorStatus()
        self._settings = NLPDifferentiatorSettings(**kwargs)

        self._prepare_differentiator()

    @property
    def status(self) -> NLPDifferentiatorStatus:
        """
        Status of the NLP differentiator. This is an annotated dataclass that can also be printed for convenience.
        See :py:class:`do_mpc.differentiator.helper.NLPDifferentiatorStatus` for more information.
        """
        return self._status
    
    @property
    def settings(self) -> NLPDifferentiatorSettings:
        """
        Settings of the NLP differentiator. This is an annotated dataclass that can also be printed for convenience.
        See :py:class:`do_mpc.differentiator.helper.NLPDifferentiatorSettings` for more information.

        **Example**:

        ::

            nlp_diff = NLPDifferentiator(nlp, nlp_bounds)
            nlp_diff.settings.check_licq = False

        Note:
            Settings can also be passed as keyword arguments to the constructor of :py:class:`NLPDifferentiator`.
        """
        return self._settings

    def _prepare_differentiator(self):
        """
        Warning:
            Not part of the public API.

        This method is called in the constructor of :py:class:`NLPDifferentiator` 
        and prepares the differentiator for the differentiation of the NLP.
        """
        self._remove_unused_sym_vars()
        self._get_size_metrics()
        self._get_sym_lagrange_multipliers()
        self._stack_primal_dual()
        self._get_Lagrangian_sym()
        self._prepare_sensitivity_matrices()
        self._prepare_constraint_gradients()
        
    def _detect_undetermined_sym_var(self, var: str ="x") -> Tuple[np.ndarray,np.ndarray]: 
        
        # symbolic expressions
        var_sym = self.nlp[var]        
        # objective function
        f_sym = self.nlp["f"]
        # constraints
        g_sym = self.nlp["g"]

        # boolean expressions on wether a symbolic is contained in the objective function f or the constraints g
        map_f_var = map(lambda x: ca.depends_on(f_sym,x),ca.vertsplit(var_sym))
        map_g_var = map(lambda x: ca.depends_on(g_sym,x),ca.vertsplit(var_sym))

        # combined boolean expressions as list for each symbolic variable in var_sym
        dep_list = [f_dep or g_dep for f_dep,g_dep in zip(map_f_var,map_g_var)]

        # indices of undetermined and determined symbolic variables
        undet_sym_idx = np.where(np.logical_not(dep_list))[0]
        det_sym_idx = np.where(dep_list)[0]

        return undet_sym_idx, det_sym_idx

    def _remove_unused_sym_vars(self):
        """
        Warning:
            Not part of the public API.

        Reduces the NLP by removing symbolic variables 
        for x and p that are not contained in the objective function or the constraints.

        """
        # detect undetermined symbolic variables
        undet_opt_x_idx, det_opt_x_idx = self._detect_undetermined_sym_var("x")
        undet_opt_p_idx, det_opt_p_idx = self._detect_undetermined_sym_var("p")
        
        # copy nlp and nlp_bounds
        nlp_red = self.nlp.copy()
        nlp_bounds_red = self.nlp_bounds.copy()

        # adapt nlp
        nlp_red["x"] = self.nlp["x"][det_opt_x_idx]
        nlp_red["p"] = self.nlp["p"][det_opt_p_idx]

        # adapt nlp_bounds
        nlp_bounds_red["lbx"] = self.nlp_bounds["lbx"][det_opt_x_idx]
        nlp_bounds_red["ubx"] = self.nlp_bounds["ubx"][det_opt_x_idx]

        det_sym_idx_dict = {"opt_x":det_opt_x_idx, "opt_p":det_opt_p_idx}
        undet_sym_idx_dict = {"opt_x":undet_opt_x_idx, "opt_p":undet_opt_p_idx}

        N_vars_to_remove = len(undet_sym_idx_dict["opt_x"])+len(undet_sym_idx_dict["opt_p"])
        if N_vars_to_remove > 0:
            self.nlp_unreduced, self.nlp_bounds_unreduced = self.nlp, self.nlp_bounds
            self.nlp, self.nlp_bounds = nlp_red, nlp_bounds_red
            self.det_sym_idx_dict, self.undet_sym_idx_dict = det_sym_idx_dict, undet_sym_idx_dict
            self.status.reduced_nlp = True
        else:
            self.status.reduced_nlp = False
            print("NLP formulation does not contain unused variables.")

    def _get_size_metrics(self):
        """
        Warning:
            Not part of the public API.

        Specifies the number of decision variables, nonlinear constraints and parameters of the NLP.
        """
        self.n_x = self.nlp["x"].shape[0]
        self.n_g = self.nlp["g"].shape[0]
        self.n_p = self.nlp["p"].shape[0]

        if self.status.reduced_nlp:
            self.n_x_unreduced = self.nlp_unreduced["x"].shape[0]
            self.n_p_unreduced = self.nlp_unreduced["p"].shape[0]

    def _get_sym_lagrange_multipliers(self):
        """
        Warning:
            Not part of the public API.

        Adds symbolic variables for the Lagrange multipliers to the NLP.
        """
        self.nlp["lam_g"] = ca.SX.sym("lam_g",self.n_g,1)
        self.nlp["lam_x"] = ca.SX.sym("lam_x",self.n_x,1)
        self.nlp["lam"] = ca.vertcat(self.nlp["lam_g"],self.nlp["lam_x"])

    def _stack_primal_dual(self):
        """
        Warning:
            Not part of the public API.

        Stacks the primal and dual variables of the NLP.
        """

        self.nlp["z"] = ca.vertcat(self.nlp["x"],self.nlp["lam"])

    def _get_Lagrangian_sym(self): 
        """
        Warning:
            Not part of the public API.

        Sets the Lagrangian of the NLP for sensitivity calculation.
        Attention: It is not verified, whether the NLP is in standard form. 
        """
        # TODO: verify if NLP is in standard form to simplify further evaluations
        self.L_sym = self.nlp["f"] + self.nlp['lam_g'].T @ self.nlp['g'] + self.nlp['lam_x'].T @ self.nlp['x']

    def _prepare_sensitivity_matrices(self):
        """
        Warning:
            Not part of the public API.

        Calculates the sensitivity matrices of the NLP.
        """
        self.A_sym = ca.hessian(self.L_sym,self.nlp["z"])[0]
        self.A_func = ca.Function("A", [self.nlp["z"],self.nlp["p"]], [self.A_sym], ["z_opt", "p_opt"], ["A"])

        # TODO: Note, full parameter vector considered for differentiation. This is not necessary, if only a subset of the parametric sensitivities is required. Future version will considere reduces parameter space.
        self.B_sym = ca.jacobian(ca.gradient(self.L_sym,self.nlp["z"]),self.nlp["p"])
        self.B_func = ca.Function("B", [self.nlp["z"],self.nlp["p"]], [self.B_sym], ["z_opt", "p_opt"], ["B"])

        self.status.sym_KKT = True


    def _prepare_constraint_gradients(self):
        """
        Warning:
            Not part of the public API.

        Calculates the gradients of the constraints of the NLP.
        """
        self.cons_sym = ca.vertcat(self.nlp["g"],self.nlp["x"])
        self.cons_grad_sym = ca.jacobian(self.cons_sym,self.nlp["x"])
        self.cons_grad_func = ca.Function("cons_grad", [self.nlp["x"],self.nlp["p"]], [self.cons_grad_sym], ["x_opt", "p_opt"], ["d(g,x)/dx"])

    def _reduce_nlp_solution_to_determined(self, nlp_sol: Dict, p_num: ca.DM) -> Tuple[dict, ca.DM]: 
        """
        Warning:
            Not part of the public API.
        
        Maps the full NLP solutions to the reduced NLP solutions (the determined variables).

        Args:
            nlp_sol: Full NLP solution.
            p_num: Numerical parameter vector.

        Returns:
            Reduced NLP solution ``nlp_sol_red`` and Reduced parameter vector ``p_num``.

        """

        assert self.status.reduced_nlp, "NLP is not reduced."

        # adapt nlp_sol
        nlp_sol_red = nlp_sol.copy()
        nlp_sol_red["x"] = nlp_sol["x"][self.det_sym_idx_dict["opt_x"]]
        nlp_sol_red["lam_x"] = nlp_sol["lam_x"][self.det_sym_idx_dict["opt_x"]] 
        p_num = p_num[self.det_sym_idx_dict["opt_p"]]
        
        # backwards compatilibity TODO: remove
        if "x_unscaled" in nlp_sol:
            nlp_sol_red["x_unscaled"] = nlp_sol["x_unscaled"][self.det_sym_idx_dict["opt_x"]]

        return nlp_sol_red, p_num
    
    def _get_active_constraints(self, nlp_sol: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Warning:
            Not part of the public API.

        This function determines the active set of the current NLP solution. 
        The active set is determined by the "primal" solution, 
        considering the bounds on the variables and constraints.
        The active set is returned as a list of numpy arrays 
        containing the indices of the active and inactive nonlinear and linear constraints.

        Args:
            nlp_sol: the NLP solution.

        Returns:
            where_g_inactive: Indices of the inactive nonlinear constraints.
            where_x_inactive: Indices of the inactive linear constraints.
            where_g_active: Indices of the active nonlinear constraints.
            where_x_active: Indices of the active linear constraints.

        Raises:
            KeyError: If the NLP solution does not contain the primal or dual solution.
        """


        x_num = nlp_sol["x"]
        g_num = nlp_sol["g"]
        
        # determine active set
        ## bounds of nonlinear and linear constraints
        lbg = self.nlp_bounds["lbg"]
        ubg = self.nlp_bounds["ubg"]
        lbx = self.nlp_bounds["lbx"]
        ubx = self.nlp_bounds["ubx"]

        ## determine distance to bounds
        g_delta_lbg = g_num - lbg
        g_delta_ubg = g_num - ubg
        x_delta_lbx = x_num - lbx
        x_delta_ubx = x_num - ubx

        ## determine active set based on distance to bounds with tolerance tol
        where_g_inactive = np.where((np.abs(g_delta_lbg)>self.settings.active_set_tol) & (np.abs(g_delta_ubg)>self.settings.active_set_tol))[0]
        where_x_inactive = np.where((np.abs(x_delta_lbx)>self.settings.active_set_tol) & (np.abs(x_delta_ubx)>self.settings.active_set_tol))[0]            
        where_g_active = np.where((np.abs(g_delta_lbg)<=self.settings.active_set_tol) | (np.abs(g_delta_ubg)<=self.settings.active_set_tol))[0]
        where_x_active = np.where((np.abs(x_delta_lbx)<=self.settings.active_set_tol) | (np.abs(x_delta_ubx)<=self.settings.active_set_tol))[0]
        
        return where_g_inactive, where_x_inactive, where_g_active, where_x_active 
    
    def _extract_active_primal_dual_solution(self, nlp_sol: Dict) -> Tuple[ca.DM,np.ndarray]:
        """
        Warning:
            Not part of the public API.


        This function extracts the active primal and dual solution from the NLP solution and stackes it into a single vector. 
        The active set is determined by the "primal" or "dual" solution.
        Lagrange multipliers of inactive constraints can be set to zero with the argument set_lam_zero.
        
        Args:
            nlp_sol: the NLP solution.

        Returns:
            z_num: casadi DM containing the active primal and dual solution.
            where_cons_active: numpy array containing the indices of the active constraints.
        """


        where_g_inactive, where_x_inactive, where_g_active, where_x_active = self._get_active_constraints(nlp_sol)
        
        where_cons_active = np.concatenate((where_g_active,where_x_active+self.n_g))
        where_cons_inactive = np.concatenate((where_g_inactive,where_x_inactive+self.n_g))

        # set lagrange multipliers of inactive constraints to zero
        if self.settings.set_lam_zero:
            lam_num[where_cons_inactive] = 0
        
        # stack primal and dual solution
        x_num = nlp_sol["x"]
        lam_num = ca.vertcat(nlp_sol["lam_g"],nlp_sol["lam_x"])
        z_num = ca.vertcat(x_num,lam_num)

        return z_num, where_cons_active        
    
    def _get_sensitivity_matrices(self, z_num: ca.DM, p_num: ca.DM) -> Tuple[ca.DM]:
        """
        Warning:
            Not part of the public API.

        Args:
            z_num: Stacked vector of the primal and dual solution.
            p_num: Parameter vector.
        """
        if self.status.sym_KKT is False:
            raise RuntimeError('No symbolic expression for sensitivitiy system computed yet.')        

        A_num = self.A_func(z_num, p_num)
        B_num = self.B_func(z_num, p_num)
        return A_num, B_num
    
    def _reduce_sensitivity_matrices(self, A_num: ca.DM, B_num: ca.DM, where_cons_active: np.ndarray) -> Tuple[ca.DM, ca.DM]:
        """
        Warning:
            Not part of the public API.

        Reduces the sensitivity matrix A and the sensitivity vector B 
        of the NLP such that only the rows and columns corresponding to non-zero dual variables are kept.

        Args:
            A_num: Full A-Matrix
            B_num: Full B-Matrix

        Returns:
            A_num: Reduced A-Matrix
            B_num: Reduced B-Matrix

        """
        where_keep_idx = [i for i in range(self.n_x)]+list(where_cons_active+self.n_x)
        A_num = A_num[where_keep_idx,where_keep_idx]
        B_num = B_num[where_keep_idx,:]
        return A_num, B_num

    def _solve_linear_system(self,A_num: ca.DM,B_num: ca.DM, lin_solver=None) -> np.ndarray:
        """
        Solves the linear system of equations to calculate parametric sensitivities.

        Args:
            A_num: Numeric A-Matrix ``(dF/dz)``.
            B_num: Numeric B-Matrix ``(dF/dp)``.
            lin_solver: Linear solver to use. Options are ``scipy``, ``casadi`` and ``lstq``.

        Returns:
            parametric sensitivities of shape ``(n_x,n_p)``
        """
        self.status.lse_solved = False

        try:
            if lin_solver == 'scipy':
                logging.info("Solving linear system with Scipy.")
                param_sens = sp_sparse.linalg.spsolve(A_num.tocsc(),-B_num.tocsc())

            elif lin_solver == 'casadi':
                logging.info("Solving linear system with Casadi.")
                ca_lin_solver = ca.Linsol("sol","qr",A_num.sparsity())
                param_sens = ca_lin_solver.solve(A_num, -B_num)

            elif lin_solver ==  'lstsq':
                logging.info("Solving linear system with least squares solver.")
                param_sens = np.linalg.lstsq(A_num, -B_num, rcond=None)[0]

            else:
                raise Exception("Linear solver not recognized.")

            self.status.lse_solved = True

        except Exception as e:
            logging.exception(e)
            logging.info("Linear system could not be solved. Return array of NaNs for parametric sensitivities.")
            param_sens = np.full(shape=(A_num.shape[0], B_num.shape[1]), fill_value=np.nan)

            self.status.lse_solved = False

        return param_sens
    
    def _calculate_sensitivities(self, z_num: ca.DM, p_num: ca.DM, where_cons_active: np.ndarray) -> np.ndarray:
        """
        Calculates the sensitivities of the NLP solution.

        Args:
            nlp_sol: dict containing the NLP solution.
            method_active_set: str, either "primal" or "dual". Determines the active set by the primal or dual solution.
            tol: float, tolerance for determining the active set.

        Returns:
            parametric sensitivities of shape ``(n_x,n_p)``
        """

        if self.settings.check_LICQ:
            LICQ_status = self._check_LICQ(z_num[:self.n_x], p_num,where_cons_active)
            self.status.LICQ = LICQ_status
            logging.info('LICQ status: {}'.format(LICQ_status))
        
        if self.settings.check_SC:
            SC_status = self._check_SC(z_num[self.n_x:], where_cons_active)
            self.status.SC = SC_status
            logging.info('SC status: {}'.format(SC_status))
        
        A_num, B_num = self._get_sensitivity_matrices(z_num, p_num)
        A_num, B_num = self._reduce_sensitivity_matrices(A_num, B_num, where_cons_active)

        if self.settings.check_rank:
            full_rank_status =  self._check_rank(A_num)
            self.status.full_rank = full_rank_status
            logging.info('Full rank status: {}'.format(full_rank_status))

        # solve LSE to get parametric sensitivities
        param_sens = self._solve_linear_system(A_num,B_num, lin_solver=self.settings.lin_solver)

        if not self.status.lse_solved and self.settings.lstsq_fallback:
            logging.info('Linear system could not be solved. Falling back to least squares solver.')
            param_sens = self._solve_linear_system(A_num,B_num, lin_solver="lstsq")

                        
        if self.settings.track_residuals:
            residuals = self._track_residuals(A_num, B_num, param_sens)
            self.status.residuals = residuals
            
        return param_sens
    
    ### Mapping functions ###
    def _map_dxdp(self, param_sens: np.ndarray) -> np.ndarray:
        """
        Maps the parametric sensitivities to the original decision variables.
        """
        if sp_sparse.issparse(param_sens):
            dx_dp = param_sens[:self.n_x,:].toarray()
        else:
            dx_dp = param_sens[:self.n_x,:]
        return dx_dp
    
    def _map_dlamdp(self, param_sens: np.ndarray, where_cons_active: np.ndarray) -> np.ndarray:
        """
        Maps the parametric sensitivities to the original sensitivities of the lagrange multipliers.
        """
        dlam_dp = np.zeros((self.n_g+self.n_x,self.n_p))
        assert len(where_cons_active) == param_sens.shape[0]-self.n_x, "Number of non-zero dual variables does not match number of parametric sensitivities for lagrange multipliers."
        
        if sp_sparse.issparse(param_sens):
            dlam_dp[where_cons_active,:] = param_sens[self.n_x:,:].toarray()
        else:
            dlam_dp[where_cons_active,:] = param_sens[self.n_x:,:]

        return dlam_dp
    
    def _map_param_sens(self, param_sens: np.ndarray, where_cons_active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps the parametric sensitivities to the original decision variables and lagrange multipliers.
        """
        dx_dp = self._map_dxdp(param_sens)
        dlam_dp = self._map_dlamdp(param_sens, where_cons_active)
        return dx_dp, dlam_dp

    def _map_param_sens_to_full(self, dx_dp_num_red: ca.DM, dlam_dp_num_red: ca.DM) -> Tuple[ca.DM, ca.DM]:
        """
        Maps the reduced parametric sensitivities to the full decision variables.
        """
        idx_x_determined, idx_p_determined = self.det_sym_idx_dict["opt_x"], self.det_sym_idx_dict["opt_p"]

        dx_dp_num = sp_sparse.lil_matrix((self.n_x_unreduced,self.n_p_unreduced))
        dx_dp_num[idx_x_determined[:,None],idx_p_determined] = dx_dp_num_red
        
        dlam_dp_num = sp_sparse.lil_matrix((self.n_g+self.n_x_unreduced,self.n_p_unreduced))

        idx_lam_determined = np.hstack([np.arange(0,self.n_g,dtype=np.int64),idx_x_determined+self.n_g])

        dlam_dp_num[idx_lam_determined[:,None],idx_p_determined] = dlam_dp_num_red

        return dx_dp_num, dlam_dp_num
    
    def _check_rank(self, A_num: ca.DM):
        """
        Checks if the sensitivity matrix A has full rank.
        """
        full_rank = np.linalg.matrix_rank(A_num) == A_num.shape[0]
        return full_rank 
    
    def _track_residuals(self, A_num: ca.DM, B_num: ca.DM, param_sens: np.ndarray) -> float:
        """
        Tracks the residuals of the linear system of equations.
        """
        residuals = ca.norm_fro(A_num@param_sens+B_num)
        return residuals

    def _check_LICQ(self, x_num: ca.DM, p_num: ca.DM, where_cons_active: np.ndarray) -> bool:        
        """
        Checks if the linear independence constraint qualification is satisfied.
        """
        cons_grad_num = self.cons_grad_func(x_num, p_num)
        cons_grad_num = cons_grad_num[where_cons_active,:]

        if cons_grad_num.shape[0] == 0:
            return True
        elif np.linalg.matrix_rank(cons_grad_num) < cons_grad_num.shape[0]:
            return False
        else:
            return True
    
    def _check_SC(self, lam_num: ca.DM, where_cons_active: np.ndarray):
        """
        Checks if the strict complementarity is satisfied.
        """

        # lagrange multipliers for active set
        lam_num = lam_num[where_cons_active]
        # check if all absolute values of lagrange multipliers are strictly greater than active set tolerance
        if np.all(np.abs(lam_num) >= self.settings.active_set_tol):
            logging.info("Strict complementarity satisfied.")
            return True
        else:
            logging.info("Strict complementarity not satisfied.")
            return False


    # the next function applies the whole algorithm given in the code abouve and returns the sensitivities dx_dp
    def differentiate(self, nlp_sol: dict, p_num: ca.DM) -> Tuple[ca.DM, ca.DM]:
        """
        Main method of the class. Call this method to obtain the parametric sensitivities.
        The sensitivity matrix ``dx_dp`` is of shape ``(n_x, n_p)``.
        
        Note:
            Please read the documentation of the class :py:class:`NLPDifferentiator` for more information.

        Args:
            nlp_sol: Dictionary containing the optimal solution of the NLP.
            p_num: Numerical value of the parameters of the NLP.

        Returns:
            Parametric sensitivities of the decision variables and lagrange multipliers.
        """

        nlp_sol_mandatory_keys = ['x', 'lam_g', 'lam_x', 'g']
        
        if not isinstance(nlp_sol, dict):
            raise ValueError('nlp_sol must be a dictionary.')

        if not set(nlp_sol.keys()).issuperset(set(nlp_sol_mandatory_keys)):
            raise ValueError('nlp_sol must contain keys {}.'.format(nlp_sol_mandatory_keys))

        if isinstance(p_num, (float, int)):
            p_num = ca.DM(p_num)
        elif isinstance(p_num, np.ndarray):
            p_num = ca.DM(p_num)
        elif isinstance(p_num, ca.DM):
            pass
        else:
            raise ValueError('p_num must be a float, int, np.ndarray or DM object. You have {}'.format(type(p_num)))
        
        # reduce NLP solution if necessary
        if self.status.reduced_nlp:
            nlp_sol, p_num = self._reduce_nlp_solution_to_determined(nlp_sol, p_num)

        if not p_num.shape == (self.n_p, 1):
            raise ValueError('p_num must have length {}.'.format(self.n_p))

        # extract active primal and dual solution
        z_num, where_cons_active = self._extract_active_primal_dual_solution(nlp_sol)

        # calculate parametric sensitivities
        param_sens = self._calculate_sensitivities(z_num, p_num, where_cons_active)

        # map sensitivities to original decision variables and lagrange multipliers
        
        dx_dp_num_red, dlam_dp_num_red = self._map_param_sens(param_sens, where_cons_active)
        if self.status.reduced_nlp:
            dx_dp_num, dlam_dp_num = self._map_param_sens_to_full(dx_dp_num_red,dlam_dp_num_red)
        else:
            dx_dp_num = dx_dp_num_red
            dlam_dp_num = dlam_dp_num_red

        return dx_dp_num, dlam_dp_num


  
class DoMPCDifferentiator(NLPDifferentiator):
    """
    Nonlinear program (NLP) Differentiator for ``do_mpc`` objects. 
    Can be used with :py:class:`do_mpc.controller.MPC` and :py:class:`do_mpc.estimator.MHE` objects.
    The class inherits the :py:class:`NLPDifferentiator` class and overwrites the :py:meth:`differentiate` method.

    **Example:**

    1. Setup a ``do_mpc`` optimizer object (e.g. :py:class:`do_mpc.controller.MPC` or :py:class:`do_mpc.estimator.MHE`).

    ::

        model = ...
        mpc = do_mpc.controller.MPC(model)
        ...
        mpc.setup()

    2. Initialize the differentiator with the ``do_mpc`` optimizer object. 

    ::

        nlp_diff = DoMPCDifferentiator(mpc)

    3. Configure the differentiator settings with the :py:attr:`settings` attribute.

    ::

        nlp_diff.settings.check_LICQ = False

    4. Solve the NLP of the original ``do_mpc`` optimizer object.

    ::

        mpc.make_step(x0)

    5. Call the :py:meth:`differentiate` method of the differentiator object to compute the parametric sensitivities at the current optimal solution previously calculated with :py:meth:`make_step()`. The current parameters and optimal solution are read from the ``do_mpc`` optimizer object.

    ::

        dx_dp_num, dlam_dp_num = nlp_diff.differentiate()

    6. Typically, we are interested in specific segments of the parametric sensitivities. These can be retrieved by powerindexing the :py:attr:`sens_num` attribute.

    ::

        du0dx0 = nlp_diff.sens_num['dxdp', indexf['_u', 0, 0], indexf['_x', 0, 0]]

    This last step returns the parametric sensitivity of the first input with respect to the initial state.


    Args:
        optimizer: ``do_mpc`` class that inherits the :py:class:`Optimizer` class, that is, a :py:class:`do_mpc.controller.MPC` or :py:class:`do_mpc.estimator.MHE` object.
    """
    def __init__(self, optimizer: Optimizer, **kwargs):
        self.optimizer = optimizer
        self.x_scaling_factors = self.optimizer.opt_x_scaling.master
        self._init_sens_sym_struct()

        nlp, nlp_bounds = self._get_do_mpc_nlp()        
        super().__init__(nlp, nlp_bounds, **kwargs)

    @property
    def sens_num(self):
        """
        The sensitivity structure of the NLP. 
        This can be queried as follows:

        ::

            from casadi.tools import indexf

            du0dx0 = nlp_diff.sens_num['dxdp', indexf['_u', 0, 0], indexf['_x0']]

        The powerindices passed to ``indexf`` are derived from the attributes:

        - :py:attr:`do_mpc.controller.MPC.opt_x`
        - :py:attr:`do_mpc.controller.MPC.opt_p`
        
        """
        return self._sens_num
    

    def _get_do_mpc_nlp(self):
        """
        Warning:
            Not part of the public API.

        This function is used to extract the symbolic expressions and bounds of the underlying NLP of the MPC.
        It is used to initialize the NLPDifferentiator class.
        """

        # 1 get symbolic expressions of NLP
        nlp = {'x': ca.vertcat(self.optimizer.opt_x), 'f': self.optimizer.nlp_obj, 'g': self.optimizer.nlp_cons, 'p': ca.vertcat(self.optimizer.opt_p)}

        # 2 extract bounds
        nlp_bounds = {}
        nlp_bounds['lbg'] = self.optimizer.nlp_cons_lb
        nlp_bounds['ubg'] = self.optimizer.nlp_cons_ub
        nlp_bounds['lbx'] = ca.vertcat(self.optimizer._lb_opt_x)
        nlp_bounds['ubx'] = ca.vertcat(self.optimizer._ub_opt_x)

        return nlp, nlp_bounds

    def _get_do_mpc_nlp_sol(self):
        """
        Warning:
            Not part of the public API.

        Reads the optimal solution of the underlying NLP of the MPC.
        """

        if not hasattr(self, 'nlp_sol'):
            self.nlp_sol = {}

        self.nlp_sol["x"] = ca.vertcat(self.optimizer.opt_x_num)
        self.nlp_sol["x_unscaled"] = ca.vertcat(self.optimizer.opt_x_num_unscaled)
        self.nlp_sol["g"] = ca.vertcat(self.optimizer.opt_g_num)
        self.nlp_sol["lam_g"] = ca.vertcat(self.optimizer.lam_g_num)
        self.nlp_sol["lam_x"] = ca.vertcat(self.optimizer.lam_x_num)

        return self.nlp_sol

    def _get_p_num(self):
        """
        Warning:
            Not part of the public API.

        Reads the current parameters of the underlying NLP of the MPC.

        """
    
        return ca.vertcat(self.optimizer.opt_p_num)
    
    def differentiate(self):
        """
        Main method of the class. Computes the parametric sensitivities of the underlying NLP of the MPC or MHE.
        Should be called after solving the underlying NLP.
        The current optimal solution and the corresponding parameters are read from the ``do_mpc`` object.  
        """

        nlp_sol = self._get_do_mpc_nlp_sol()
        p_num = self._get_p_num()
        dx_dp_num, dlam_dp_num = super().differentiate(nlp_sol, p_num)
        
        # rescale dx_dp_num
        dx_dp_num = ca.times(dx_dp_num,self.x_scaling_factors.tocsc())

        # Set values on sens_num
        self.sens_num["dxdp"] = dx_dp_num

        return dx_dp_num, dlam_dp_num
    
    def _init_sens_sym_struct(self):
        opt_x = self.optimizer._opt_x
        opt_p = self.optimizer._opt_p
        
        sens_struct = castools.struct_symSX([
            castools.entry("dxdp",shapestruct=(opt_x, opt_p)),
        ])

        self._sens_num = sens_struct(0)
