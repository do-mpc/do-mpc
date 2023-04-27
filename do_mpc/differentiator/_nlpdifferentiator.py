import numpy as np
import scipy.linalg as sp_linalg
import scipy.sparse as sp_sparse
from casadi import *
from casadi.tools import *
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional, Any

from do_mpc.optimizer import Optimizer

from functools import wraps
import time
import warnings
# set np.linalg.LinalgWarning to np.linalg.LinalgError
warnings.filterwarnings("error",category=sp_linalg.LinAlgWarning)

### Helper Functions NLP Differentiator
# def build_sens_sym_struct(mpc: Optimizer):
#     opt_x = mpc._opt_x
#     opt_p = mpc._opt_p
    
#     sens_struct = struct_symSX([
#         entry("dxdp",shapestruct=(opt_x, opt_p)),
#     ])

#     return sens_struct

# def assign_num_to_sens_struct(sens_struct,dxdp_num,undet_sym_idx_dict):

#     dxdp_init = dxdp_num#.copy()
#     ins_idx_x = [val-idx for idx, val in enumerate(undet_sym_idx_dict["opt_x"])] # used for inserting zero rows in dxdp_init
#     ins_idx_p = [val-idx for idx, val in enumerate(undet_sym_idx_dict["opt_p"])] # used for inserting zero columns in dxdp_init
    
#     dxdp_init = np.insert(dxdp_init, ins_idx_x, 0.0, axis=0)
#     dxdp_init = np.insert(dxdp_init, ins_idx_p, 0.0, axis=1)
    
#     assert dxdp_init.shape == sens_struct["dxdp"].shape
    
#     sens_num = sens_struct(0)
    
#     sens_num["dxdp"] = dxdp_init

#     return sens_num


@dataclass
class NLPDifferentiatorSettings:
    """Settings for NLPDifferentiator.
    """

    lin_solver: str = 'casadi' #'scipy' or 'casadi' or 'lstsq'
    """
    Choose the linear solver for the KKT system.
    """

    check_LICQ: bool = True
    """
    Check if the constraints are linearly independent.    
    """

    check_SC: bool = True
    """
    Check if strict complementarity holds.   
    """

    track_residuals: bool = True
    """
    Compute the residuals of the KKT system.
    """

    check_rank: bool = False
    """
    ....
    """

    lstsq_fallback: bool = False
    """
    ...
    """

    active_set_tol : float = 1e-6
    """
    ...
    """

    set_lam_zero: bool = False
    """
    ...
    """


### NLP Differentiator
class NLPDifferentiator:
    """
    Documentation for NLPDifferentiator.
 
    .. warning::

        This tool is currently not fully implemented and cannot be used.
    """

    def __init__(self, nlp_container, **kwargs):
        
        ## Setup
        self._setup_nlp(nlp_container)

        self.flags = {}
        self.flags['sym_KKT_system'] = False
        self.flags['reduced_nlp'] = False

        self.settings = NLPDifferentiatorSettings(**kwargs)

        ## Preparation
        self._prepare_differentiator()
                
    ### SETUP
    def _setup_nlp(self, nlp_container: dict) -> None:
        if isinstance(nlp_container, dict):
            self.nlp, self.nlp_bounds = nlp_container["nlp"].copy(), nlp_container["nlp_bounds"].copy()
        else:
            raise ValueError('nlp_container must be a dictionary with keys "nlp" and "nlp_bounds".')

    ### PREPARATION
    def _prepare_differentiator(self):
        # 1. Detect undetermined symbolic variables and reduce NLP
        # if self.flags['reduced_nlp']:
        self._remove_unused_sym_vars()

        # 2. Get size metrics
        self._get_size_metrics()

        # 3. Get symbolic expressions for lagrange multipliers
        self._get_sym_lagrange_multipliers()
        self._stack_primal_dual()

        # 4. Get symbolic expressions for Lagrangian
        self._get_Lagrangian_sym()
        
        # 5. Get symbolic expressions for sensitivity matrices
        self._prepare_sensitivity_matrices()

        # 6. Prepare gradient d(g,x)/dx
        self._prepare_constraint_gradients()
        
    def _detect_undetermined_sym_var(self, var: str ="x") -> tuple[np.ndarray,np.ndarray]: #TODO: change data structure of return to tuple of lists (beware that code might break)
        
        # symbolic expressions
        var_sym = self.nlp[var]        
        # objective function
        f_sym = self.nlp["f"]
        # constraints
        g_sym = self.nlp["g"]

        # boolean expressions on wether a symbolic is contained in the objective function f or the constraints g
        map_f_var = map(lambda x: depends_on(f_sym,x),vertsplit(var_sym))
        map_g_var = map(lambda x: depends_on(g_sym,x),vertsplit(var_sym))

        # combined boolean expressions as list for each symbolic variable in var_sym
        dep_list = [f_dep or g_dep for f_dep,g_dep in zip(map_f_var,map_g_var)]

        # indices of undetermined and determined symbolic variables
        undet_sym_idx = np.where(np.logical_not(dep_list))[0]
        det_sym_idx = np.where(dep_list)[0]

        # example:
        # if undet_sym_idx = [1,3], then the second and fourth symbolic variable in var_sym are undetermined
                
        return undet_sym_idx,det_sym_idx

    def _remove_unused_sym_vars(self):
        """
        Reduces the NLP by removing symbolic variables for x and p that are not contained in the objective function or the constraints.

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
            self.flags["reduced_nlp"] = True
            # self.flags["fully_determined_nlp"] = True
        else:
            self.flags["reduced_nlp"] = False
            # self.flags["fully_determined_nlp"] = True
            print("NLP formulation does not contain unused variables.")

    def _get_size_metrics(self):
        """
        Specifies the number of decision variables, nonlinear constraints and parameters of the NLP.
        """
        self.n_x = self.nlp["x"].shape[0]
        self.n_g = self.nlp["g"].shape[0]
        self.n_p = self.nlp["p"].shape[0]

        if self.flags["reduced_nlp"]:
            self.n_x_unreduced = self.nlp_unreduced["x"].shape[0]
            self.n_p_unreduced = self.nlp_unreduced["p"].shape[0]

    def _get_sym_lagrange_multipliers(self):
        self.nlp["lam_g"] = SX.sym("lam_g",self.n_g,1)
        self.nlp["lam_x"] = SX.sym("lam_x",self.n_x,1)
        self.nlp["lam"] = vertcat(self.nlp["lam_g"],self.nlp["lam_x"])

    def _stack_primal_dual(self):
        self.nlp["z"] = vertcat(self.nlp["x"],self.nlp["lam"])

    def _get_Lagrangian_sym(self): 
        """
        Sets the Lagrangian of the NLP for sensitivity calculation.
        Attention: It is not verified, whether the NLP is in standard form. 

        """
        # TODO: verify if NLP is in standard form to simplify further evaluations
        self.L_sym = self.nlp["f"] + self.nlp['lam_g'].T @ self.nlp['g'] + self.nlp['lam_x'].T @ self.nlp['x']
        # self.flags['get_Lagrangian'] = True

    def _get_A_matrix(self):
        self.A_sym = hessian(self.L_sym,self.nlp["z"])[0]
        self.A_func = Function("A", [self.nlp["z"],self.nlp["p"]], [self.A_sym], ["z_opt", "p_opt"], ["A"])

    def _get_B_matrix(self):
        # TODO: Note, full parameter vector considered for differentiation. This is not necessary, if only a subset of the parametric sensitivities is required. Future version will considere reduces parameter space.
        self.B_sym = jacobian(gradient(self.L_sym,self.nlp["z"]),self.nlp["p"])
        self.B_func = Function("B", [self.nlp["z"],self.nlp["p"]], [self.B_sym], ["z_opt", "p_opt"], ["B"])

    def _prepare_sensitivity_matrices(self):
        self._get_A_matrix()
        self._get_B_matrix()
        self.flags['sym_KKT_system'] = True

    def _prepare_constraint_gradients(self):
        self.cons_sym = vertcat(self.nlp["g"],self.nlp["x"])
        self.cons_grad_sym = jacobian(self.cons_sym,self.nlp["x"])
        self.cons_grad_func = Function("cons_grad", [self.nlp["x"],self.nlp["p"]], [self.cons_grad_sym], ["x_opt", "p_opt"], ["d(g,x)/dx"])

    ### ALGORITHM    
    def _reduce_nlp_solution_to_determined(self,nlp_sol: dict) -> dict: 
        assert self.flags["reduced_nlp"], "NLP is not reduced."

        # adapt nlp_sol
        nlp_sol_red = nlp_sol.copy()
        nlp_sol_red["x"] = nlp_sol["x"][self.det_sym_idx_dict["opt_x"]]
        nlp_sol_red["lam_x"] = nlp_sol["lam_x"][self.det_sym_idx_dict["opt_x"]] 
        nlp_sol_red["p"] = nlp_sol["p"][self.det_sym_idx_dict["opt_p"]]
        
        # backwards compatilibity TODO: remove
        if "x_unscaled" in nlp_sol:
            nlp_sol_red["x_unscaled"] = nlp_sol["x_unscaled"][self.det_sym_idx_dict["opt_x"]]

        return nlp_sol_red
    
    def _get_active_constraints(self,nlp_sol: dict) -> tuple[np.ndarray]:
        """
        This function determines the active set of the current NLP solution. The active set is determined by the "primal" solution, considering the bounds on the variables and constraints.
        The active set is returned as a list of numpy arrays containing the indices of the active and inactive nonlinear and linear constraints.

        Args:
            nlp_sol: dict containing the NLP solution.
            tol: tolerance for the active set detection. Default: 1e-6. (Should be related to optimizer tolerance.)

        Returns:
            where_g_inactive: numpy array containing the indices of the inactive nonlinear constraints.
            where_x_inactive: numpy array containing the indices of the inactive linear constraints.
            where_g_active: numpy array containing the indices of the active nonlinear constraints.
            where_x_active: numpy array containing the indices of the active linear constraints.

        Raises:
            KeyError: If the NLP solution does not contain the primal or dual solution.
        """

        if "x" not in nlp_sol.keys():
            raise KeyError("NLP solution does not contain primal solution.")
        if "lam_g" not in nlp_sol.keys():
            raise KeyError("NLP solution does not contain dual solution to nonlinear constraints.")
        if "lam_x" not in nlp_sol.keys():
            raise KeyError("NLP solution does not contain dual solution to linear constraints.")
        if "g" not in nlp_sol.keys():
            raise KeyError("NLP solution does not contain nonlinear constraints.")        
        
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
    
    def _extract_active_primal_dual_solution(self, nlp_sol: dict) -> tuple[DM,np.ndarray]:
        """
        This function extracts the active primal and dual solution from the NLP solution and stackes it into a single vector. The active set is determined by the "primal" or "dual" solution.
        Lagrange multipliers of inactive constraints can be set to zero with the argument set_lam_zero.
        
        Args:
            nlp_sol: dict containing the NLP solution.
            tol: tolerance for the active set detection. Default: 1e-6. (Should be related to optimizer tolerance.)
            set_lam_zero: bool, if True, the dual solution is set to zero for inactive constraints. Default: False.

        Returns:
            z_num: casadi DM containing the active primal and dual solution.
            where_cons_active: numpy array containing the indices of the active constraints.
        """

        x_num = nlp_sol["x"]
        lam_num = vertcat(nlp_sol["lam_g"],nlp_sol["lam_x"])

        where_g_inactive, where_x_inactive, where_g_active, where_x_active = self._get_active_constraints(nlp_sol)
        
        where_cons_active = np.concatenate((where_g_active,where_x_active+self.n_g))
        where_cons_inactive = np.concatenate((where_g_inactive,where_x_inactive+self.n_g))

        # set lagrange multipliers of inactive constraints to zero
        if self.settings.set_lam_zero:
            lam_num[where_cons_inactive] = 0
        
        # stack primal and dual solution
        z_num = vertcat(x_num,lam_num)

        return z_num, where_cons_active        
    
    def _get_sensitivity_matrices(self, z_num: DM, p_num: DM) -> tuple[DM]:
        """
        Returns the sensitivity matrix A and the sensitivity vector B of the NLP.
        """
        if self.flags['sym_KKT_system'] is False:
            raise RuntimeError('No symbolic expression for sensitivitiy system computed yet.')        
        A_num = self.A_func(z_num, p_num)
        B_num = self.B_func(z_num, p_num)
        return A_num, B_num
    
    def _reduce_sensitivity_matrices(self, A_num: DM, B_num: DM, where_cons_active: np.ndarray) -> tuple[DM]:
        """
        Reduces the sensitivity matrix A and the sensitivity vector B of the NLP such that only the rows and columns corresponding to non-zero dual variables are kept.
        """
        where_keep_idx = [i for i in range(self.n_x)]+list(where_cons_active+self.n_x)
        A_num = A_num[where_keep_idx,where_keep_idx]
        B_num = B_num[where_keep_idx,:]
        return A_num, B_num

    def _solve_linear_system(self,A_num: DM,B_num: DM, lin_solver=None) -> np.ndarray:
        """
        Solves the linear system of equations to calculate parametric sensitivities.
        Args:
            A_num: Numeric A-Matrix (dF/dz).
            B_num: Numeric B-Matrix (dF/dp).
            lin_solver: Linear solver to use. Options are "scipy", "casadi" and "lstq".
        Returns:
            parametric sensitivities (n_x,n_p)
        """
        if lin_solver == "scipy":
            with np.errstate(all="raise"):
                # param_sens = sp_linalg.solve(A_num.full(), -B_num.full(), assume_a="sym")
                param_sens = sp_sparse.linalg.spsolve(A_num.tocsc(),-B_num.tocsc())
        elif lin_solver == "casadi":
            ca_lin_solver = Linsol("sol","qr",A_num.sparsity())
            # ca_lin_solver = Linsol("sol","qr",A_num.sparsity())
            param_sens = ca_lin_solver.solve(A_num, -B_num)
            # param_sens = solve(A_num, -B_num)
        elif lin_solver == "lstsq":
            param_sens = np.linalg.lstsq(A_num, -B_num, rcond=None)[0]
        else:
            raise ValueError("Unknown linear solver.")
        return param_sens
    
    def _calculate_sensitivities(self, z_num: DM, p_num: DM, where_cons_active: np.ndarray):
        """
        Calculates the sensitivities of the NLP solution.
        Args:
            nlp_sol: dict containing the NLP solution.
            method_active_set: str, either "primal" or "dual". Determines the active set by the primal or dual solution.
            tol: float, tolerance for determining the active set.
        Returns:
        """
        # returns
        LICQ_status = None
        SC_status = None
        residuals = None

        if self.settings.check_LICQ:
            LICQ_status = self._check_LICQ(z_num[:self.n_x], p_num,where_cons_active)
        
        if self.settings.check_SC:
            SC_status = self._check_SC(z_num[self.n_x:], where_cons_active)
        
        A_num, B_num = self._get_sensitivity_matrices(z_num, p_num)
        A_num, B_num = self._reduce_sensitivity_matrices(A_num, B_num, where_cons_active)
        
        # assert np.all(sp_linalg.eigvals(A_num)>0)
        # assert where_cons_active.shape[0]<= self.n_x

        if self.settings.check_rank:
            self._check_rank(A_num)

        # solve LSE to get parametric sensitivities
        try:
            param_sens = self._solve_linear_system(A_num,B_num, lin_solver=self.settings.lin_solver)
        # except np.linalg.LinAlgError:
        except:
            if self.settings.lstsq_fallback:
                print("Solving LSE failed. Falling back to least squares solution.")
                param_sens = self._solve_linear_system(A_num,B_num, lin_solver="lstsq")
            else:
                raise np.linalg.LinAlgError("Solving LSE failed.")
                        
        if self.settings.track_residuals:
            residuals = self._track_residuals(A_num, B_num, param_sens)
            
        return param_sens, residuals, LICQ_status, SC_status
    
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
    
    def _map_param_sens(self, param_sens: np.ndarray, where_cons_active: np.ndarray) -> tuple[np.ndarray]:
        """
        Maps the parametric sensitivities to the original decision variables and lagrange multipliers.
        """
        dx_dp = self._map_dxdp(param_sens)
        dlam_dp = self._map_dlamdp(param_sens, where_cons_active)
        return dx_dp, dlam_dp

    def _map_param_sens_to_full(self, dx_dp_num_red: DM, dlam_dp_num_red: DM) -> tuple[DM]:
        """
        Maps the reduced parametric sensitivities to the full decision variables.
        """
        idx_x_determined, idx_p_determined = self.det_sym_idx_dict["opt_x"], self.det_sym_idx_dict["opt_p"]

        dx_dp_num = np.zeros((self.n_x_unreduced,self.n_p_unreduced))
        dx_dp_num = sp_sparse.csc_matrix(dx_dp_num)
        dx_dp_num[idx_x_determined[:,None],idx_p_determined] = dx_dp_num_red
        
        dlam_dp_num = np.zeros((self.n_g+self.n_x_unreduced,self.n_p_unreduced))
        dlam_dp_num = sp_sparse.csc_matrix(dlam_dp_num)

        idx_lam_determined = np.hstack([np.arange(0,self.n_g,dtype=np.int64),idx_x_determined+self.n_g])

        dlam_dp_num[idx_lam_determined[:,None],idx_p_determined] = dlam_dp_num_red

        return dx_dp_num, dlam_dp_num
    
    ### Check assumptions ###
    def _check_rank(self, A_num: DM):
        """
        Checks if the sensitivity matrix A has full rank.
        """
        if np.linalg.matrix_rank(A_num) < A_num.shape[0]:
            raise KeyError("Sensitivity matrix A does not have full rank.")
    
    def _track_residuals(self, A_num: DM, B_num: DM, param_sens: np.ndarray) -> float:
        """
        Tracks the residuals of the linear system of equations.
        """
        # residuals = np.linalg.norm(A_num.dot(param_sens)+B_num, ord=2)
        # residuals = np.linalg.norm(A_num.full().dot(param_sens)+B_num.full(), ord=2)
        residuals = norm_fro(A_num@param_sens+B_num)
        return residuals

    def _check_LICQ(self, x_num: DM, p_num: DM, where_cons_active: np.ndarray) -> bool:        
        # get constraint Jacobian
        cons_grad_num = self.cons_grad_func(x_num, p_num)
        # reduce constraint Jacobian
        cons_grad_num = cons_grad_num[where_cons_active,:]
        # check rank
        if np.linalg.matrix_rank(cons_grad_num) < cons_grad_num.shape[0]:
            # raise KeyError("Constraint Jacobian does not have full rank at current solution. LICQ not satisfied.")
            print("Constraint Jacobian does not have full rank at current solution. LICQ not satisfied.")
            return False
        else:
            print("LICQ satisfied.")
            return True
    
    def _check_SC(self, lam_num: DM, where_cons_active: np.ndarray):
        # function to check assumption of strict complementarity
        # lagrange multipliers for active set
        lam_num = lam_num[where_cons_active]
        # check if all absolute values of lagrange multipliers are strictly greater than active set tolerance
        if np.all(np.abs(lam_num) >= self.settings.active_set_tol):
            print("Strict complementarity satisfied.")
            return True
        else:
            # n_violation_SC = sum(np.abs(lam_num)<self.settings.active_set_tol)
            print("Strict complementarity not satisfied.")
            return False

    ### differentiaton step ###

    # the next function applies the whole algorithm given in the code abouve and returns the sensitivities dx_dp
    def differentiate(self, nlp_sol: dict):
        """
        Differentiates the NLP solution.
        """


        # reduce NLP solution if necessary
        if self.flags['reduced_nlp']:
            nlp_sol = self._reduce_nlp_solution_to_determined(nlp_sol)

        # get parameters of optimal solution
        p_num = nlp_sol["p"]

        # extract active primal and dual solution
        z_num, where_cons_active = self._extract_active_primal_dual_solution(nlp_sol)

        # calculate parametric sensitivities
        param_sens, residuals, LICQ_status, SC_status = self._calculate_sensitivities(z_num, p_num, where_cons_active)

        # map sensitivities to original decision variables and lagrange multipliers
        dx_dp_num_red, dlam_dp_num_red = self._map_param_sens(param_sens, where_cons_active)
        if self.flags['reduced_nlp']:
            dx_dp_num, dlam_dp_num = self._map_param_sens_to_full(dx_dp_num_red,dlam_dp_num_red)
        else:
            dx_dp_num = dx_dp_num_red

        return dx_dp_num, dlam_dp_num, residuals, LICQ_status, SC_status, where_cons_active


  
class DoMPCDifferentiatior(NLPDifferentiator):
    def __init__(self, optimizer: Optimizer, **kwargs):
        nlp_container = self._get_do_mpc_nlp(optimizer)        
        self.optimizer = optimizer
        self.x_scaling_factors = self.optimizer.opt_x_scaling.master
        super().__init__(nlp_container,**kwargs)

    def _get_do_mpc_nlp(self, mpc_object):
        """
        This function is used to extract the symbolic expressions and bounds of the underlying NLP of the MPC.
        It is used to initialize the NLPDifferentiator class.
        """

        # 1 get symbolic expressions of NLP
        nlp = {'x': vertcat(mpc_object.opt_x), 'f': mpc_object.nlp_obj, 'g': mpc_object.nlp_cons, 'p': vertcat(mpc_object.opt_p)}

        # 2 extract bounds
        nlp_bounds = {}
        nlp_bounds['lbg'] = mpc_object.nlp_cons_lb
        nlp_bounds['ubg'] = mpc_object.nlp_cons_ub
        nlp_bounds['lbx'] = vertcat(mpc_object._lb_opt_x)
        nlp_bounds['ubx'] = vertcat(mpc_object._ub_opt_x)

        # return nlp, nlp_bounds
        # return {"nlp": nlp.copy(), "nlp_bounds": nlp_bounds.copy()}
        return {"nlp": nlp, "nlp_bounds": nlp_bounds}

    def _get_do_mpc_nlp_sol(self, mpc_object):
        nlp_sol = {}
        nlp_sol["x"] = vertcat(mpc_object.opt_x_num)
        nlp_sol["x_unscaled"] = vertcat(mpc_object.opt_x_num_unscaled)
        nlp_sol["g"] = vertcat(mpc_object.opt_g_num)
        nlp_sol["lam_g"] = vertcat(mpc_object.lam_g_num)
        nlp_sol["lam_x"] = vertcat(mpc_object.lam_x_num)
        nlp_sol["p"] = vertcat(mpc_object.opt_p_num)
        return nlp_sol
    
    def differentiate(self):
        nlp_sol = self._get_do_mpc_nlp_sol(self.optimizer)
        dx_dp_num, dlam_dp_num, residuals, LICQ_status, SC_status, where_cons_active = super().differentiate(nlp_sol)
        
        # rescale dx_dp_num
        dx_dp_num = times(dx_dp_num,self.x_scaling_factors.tocsc())

        return dx_dp_num, dlam_dp_num, residuals, LICQ_status, SC_status, where_cons_active
    
    # mapping of parametric sensitivities of decision variables w.r.t parameters on casadi sym struct
    def get_dxdp_symstruct(self,dx_dp_num):
        sens_struct = self._build_sens_sym_struct(self.optimizer)
        sens_num = self._assign_num_to_sens_struct(sens_struct,dx_dp_num)
        return sens_num

    def _build_sens_sym_struct(self,mpc: Optimizer):
        opt_x = mpc._opt_x
        opt_p = mpc._opt_p
        
        sens_struct = struct_symSX([
            entry("dxdp",shapestruct=(opt_x, opt_p)),
        ])

        return sens_struct

    def _assign_num_to_sens_struct(self,sens_struct,dxdp_num):

        # dxdp_init = dxdp_num
        # ins_idx_x = [val-idx for idx, val in enumerate(self.undet_sym_idx_dict["opt_x"])] # used for inserting zero rows in dxdp_init
        # ins_idx_p = [val-idx for idx, val in enumerate(self.undet_sym_idx_dict["opt_p"])] # used for inserting zero columns in dxdp_init
        
        # dxdp_init = np.insert(dxdp_init, ins_idx_x, 0.0, axis=0)
        # dxdp_init = np.insert(dxdp_init, ins_idx_p, 0.0, axis=1)
        
        # assert dxdp_init.shape == sens_struct["dxdp"].shape
        
        sens_num = sens_struct(0)
        
        # sens_num["dxdp"] = dxdp_init
        sens_num["dxdp"] = dxdp_num

        return sens_num