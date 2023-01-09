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

class NLPHandler:
    """ 
    Documentation for NLPHandler class.

    .. warning::

        This tool is currently not fully implemented and cannot be used.

    - Transform NLP in standard form.
    - Get Lagrangian of NLP
    - Get KKT conditions
    - Get metrics about NLP (number of constraints, decision variables etc.)

    **Design principles:**
    
    - Upper bounds before lower bound
    - Inequalities before equalities
    - nonlinear constraints before linear constraints
    
    """
    def __init__(self, nlp_dict, nlp_bounds):
        self.nlp_dict = nlp_dict
        self.nlp_bounds = nlp_bounds

        self.flags = {
            'transformed':    None,
            'get_Lagrangian': False,
            'get_KKT':        False,
        }

        #TODO: Können wir die bounds noch später ändern? Ja
        #TODO: Check for nlp is already in standard form.

    def _transform_nlp_to_standard_full(self):
        """This transformation does not need any information about the current solution of the problem (e.g. active set).
        It only needs the bounds and the standard form of the problem.

        [g_nl,g_x] --> [g_nl_ubg,g_x_ubx,g_nl_lbg,g_x_lbx] + [h_nl, h_x]
        short: upper bounds before lower bounds; inequalities before equalities

        """
        # constraints:
        # 2x (ng+nx) (introduce inequality constraintes of form g(x,p)<=0)
        # 1x nh (introduce equality constraints of form h(x,p)=0)

        # 1.1 extract symbolic expressions
        x_sym = self.nlp_dict['x']
        p_sym = self.nlp_dict['p']
        f_sym = self.nlp_dict['f']
        g_sym = self.nlp_dict['g']
        lam_g_sym = SX.sym('lam_g', g_sym.shape[0])
        lam_x_sym = SX.sym('lam_x', x_sym.shape[0])

        # 1.2 extract bounds
        lbg = np.array(self.nlp_bounds['lbg'])
        ubg = np.array(self.nlp_bounds['ubg'])
        
        # TODO: Fix if only lb or only ub for x are given.
        if "lbx" in self.nlp_bounds.keys():
            lbx = np.array(self.nlp_bounds['lbx'])

        if "ubx" in self.nlp_bounds.keys():
            ubx = np.array(self.nlp_bounds['ubx'])

        # 2 nonlinear constraints (g_lb < g(x,p) < g_ub)
        is_g_equal = (lbg == ubg)
        is_g_lower = (lbg > -np.inf) *  ~is_g_equal
        is_g_upper = (ubg < np.inf)  *  ~is_g_equal

        self.where_g_equal = np.argwhere(is_g_equal)[:,0] # It was a list in the past
        self.where_g_lower = np.argwhere(is_g_lower)[:,0]
        self.where_g_upper = np.argwhere(is_g_upper)[:,0]

        nl_equal_sym = (g_sym - lbg)[self.where_g_equal]
        nl_lower_sym = (lbg - g_sym)[self.where_g_lower]
        nl_upper_sym = (g_sym - ubg)[self.where_g_upper] 

        # 2.1 detect presence of equality constraints
        lin_state_constraints_bool = ("lbx" in self.nlp_bounds.keys() and "ubx" in self.nlp_bounds.keys())

        # 2.2 state constraints (x_lb < x < x_ub)
        if lin_state_constraints_bool:
            is_x_equal = (lbx == ubx)
            is_x_lower = (lbx > -np.inf) *  ~is_x_equal
            is_x_upper = (ubx < np.inf)  *  ~is_x_equal

            self.where_x_equal = np.argwhere(is_x_equal)[:,0]
            self.where_x_lower = np.argwhere(is_x_lower)[:,0]
            self.where_x_upper = np.argwhere(is_x_upper)[:,0]

            lin_equal_sym = (x_sym - lbx)[self.where_x_equal]
            lin_lower_sym = (lbx - x_sym)[self.where_x_lower]
            lin_upper_sym = (x_sym - ubx)[self.where_x_upper]

            all_equal_sym = vertcat(nl_equal_sym, lin_equal_sym)
            all_inequal_sym = vertcat(nl_upper_sym, nl_lower_sym, lin_upper_sym, lin_lower_sym)

        else:
            all_equal_sym = nl_equal_sym
            all_inequal_sym = vertcat(nl_upper_sym, nl_lower_sym)

        # symbolically check, wether upper or lower bounds on inequality constraints are active
        sign_lam_g_sym = sign(lam_g_sym)
        sign_lam_x_sym = sign(lam_x_sym)        

        # Lagrange multiplier are positive if upper bound is active and zero if not
        # Lagrange multiplier are negative if lower bound is active and zero if not -> Invert sign
        lam_g_upper_sym =   (sign_lam_g_sym == 1) *lam_g_sym  
        lam_g_lower_sym = - (sign_lam_g_sym == -1)*lam_g_sym
        lam_x_upper_sym =   (sign_lam_x_sym == 1) *lam_x_sym
        lam_x_lower_sym = - (sign_lam_x_sym == -1)*lam_x_sym

        lam_sym_transformed = vertcat(
            lam_g_upper_sym[self.where_g_upper],
            lam_x_upper_sym[self.where_x_upper], 
            lam_g_lower_sym[self.where_g_lower], 
            lam_x_lower_sym[self.where_x_lower])

        nu_sym_transformed = vertcat(lam_g_sym[self.where_g_equal], lam_x_sym[self.where_x_equal])

        self.nu_function =  Function('nu_function',  [lam_g_sym, lam_x_sym], [nu_sym_transformed], ["lam_g", "lam_x"], ["nu_sym"])
        self.lam_function = Function('lam_function', [lam_g_sym, lam_x_sym], [lam_sym_transformed], ["lam_g", "lam_x"], ["lam_sym"])

        # 4. create full nlp
        self.nlp_standard_full_dict = {"f":f_sym, "x":x_sym, "p":p_sym, "g":all_inequal_sym, "h":all_equal_sym}
        
        # 5. change bounds
        self.n_g = all_inequal_sym.shape[0]
        self.n_h = all_equal_sym.shape[0]

        lbg_full_standard = np.concatenate((-np.inf*np.ones(self.n_g), np.zeros(self.n_h)),axis=0)
        ubg_full_standard = np.concatenate((np.zeros(self.n_g), np.zeros(self.n_h)),axis=0)

        self.nlp_standard_full_bounds = {"lbg":lbg_full_standard, "ubg":ubg_full_standard}

        # 6. symbolic expressions for dual variables
        nu_sym  = SX.sym('nu',  self.n_h) # dual variables for equality constraints
        lam_sym = SX.sym('lam', self.n_g) # dual variables for inequality constraints
        self.nlp_standard_full_dict.update({
            "lam":lam_sym,
            "nu":nu_sym
        })

        print("NLP transformed: \n")
        print("[g_nl,g_x] --> [g_nl_ubg,g_x_ubx,g_nl_lbg,g_x_lbx] + [h_nl, h_x]")

        self.flags['transformed'] = 'full_standard'


    def transform_nlp(self, variant='full_standard'):
        if variant is 'full_standard':
            self._transform_nlp_to_standard_full()
        else:
            raise NotImplementedError("Transformation variant {} is not implemented.".format(variant))

    def get_Lagrangian_sym(self):
        """
        Returns the Lagrangian of the NLP in the standard form.
        """
        if self.flags['transformed'] == 'full_standard':
            nlp = self.nlp_standard_full_dict
            self.L_sym = nlp["f"] + nlp['lam'].T @ nlp['g'] + nlp['nu'].T @ nlp['h']
        else:
            raise RuntimeError('NLP not transformed yet.')

        self.flags['get_Lagrangian'] = True

    def get_KKT_sym(self):
        raise NotImplementedError("KKT conditions not implemented yet.")

        self.KKT_full_sym = None
        self.KKT_equality_sym = None

        self.flags['get_KKT'] = True

