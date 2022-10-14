import numpy as np
from casadi import *
from casadi.tools import *
import pdb

class NLPHandler:
    """ 
    - Transform NLP in standard form.
    - Get Lagrangian of NLP
    - Get KKT conditions
    - Get metrics about NLP (number of constraints, decision variables etc.)

    Design principles:
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




class NLPDifferentiator(NLPHandler):
    """
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def prepare_hessian(self):
        if self.flags['get_Lagrangian'] is False:
            raise RuntimeError('Lagrangian not computed yet.')
        # CasADi function to compute the Hessian of the Lagrangian with opt_x and opt_lam
        hessian_inputs = [self.nlp_standard_full_dict['x'], self.nlp_standard_full_dict['lam'], self.nlp_standard_full_dict['nu']]
        hessian_sym, gradient_sym = hessian(self.L_sym, vertcat(*hessian_inputs))
        self.hessian_function = Function('hessian_function', hessian_inputs, [hessian_sym], ["x", "lam", "nu"], ["hessian"])


    def get_hessian(self, x, lam_g, lam_x, *args, **kwargs):
        # Compute Hessian with opt_x and opt_lam
        lam_num = self.lam_function(lam_g, lam_x)
        nu_num  = self.nu_function(lam_g, lam_x)

        H = self.hessian_function(x, lam_num, nu_num)

        return H

    
    def prepare_sensitivity(self):
        return
    
    def get_sensitivity(self, opt_x_num, opt_lam_num, opt_p_num):
        return

    def _get_A_matrix(self):
        return

    def _get_B_matrix(self):
        return

    

    
def setup_NLP_example_1():
    # build NLP
    # https://web.casadi.org/blog/nlp_sens/

    nlp_id = "casadi_nlp_sens_adapted"
    
    ## Decision Variables
    x_sym = SX.sym('x',2,1)

    ## Parameters
    p_sym = SX.sym('p',2,1)

    ## Objective Function
    f_sym = (p_sym[0] - x_sym[0])**2 + 0.2*(x_sym[1] - x_sym[0]**2)**2

    ## Constraint Functions
    # ubg = 0 (standard form)

    g_0 = (p_sym[1]**2)/4 - (x_sym[0]+0.5)**2 + x_sym[1]**2
    g_1 = (x_sym[0]+0.5)**2 + x_sym[1]**2- p_sym[1]**2
    g_2 = (x_sym[0]+0.5)

    # concat constraints
    g_sym = vertcat(g_0,g_1,g_2)

    ## setup NLP
    nlp = {'x':x_sym, 'p':p_sym, 'f':f_sym, 'g':g_sym}


    ## setup Bounds
    lbg = -np.inf*np.ones(g_sym.shape)
    ubg = np.zeros(g_sym.shape)
    ubg[-1] = 1.2
    lbg[-1] = 1.0

    # lbx = np.zeros(x_sym.shape)
    lbx = -np.inf*np.ones(x_sym.shape)
    ubx = np.inf*np.ones(x_sym.shape)

    # lbx[0] = 0.0
    # ubx[0] = 0.0

    nlp_bounds = {"lbg": lbg,"ubg": ubg,"lbx": lbx,"ubx": ubx}
    
    return nlp, nlp_bounds, nlp_id



def reconstruct_nlp(nlp_standard_full_dict):
    # TODO: get bounds or remove
    
    # 1. create full nlp
    f_sym = nlp_standard_full_dict["f"]
    x_sym = nlp_standard_full_dict["x"]
    p_sym = nlp_standard_full_dict["p"]
    g_sym_list = []
    if "g" in nlp_standard_full_dict.keys():
        g_sym_list.append(nlp_standard_full_dict["g"])
    if "h" in nlp_standard_full_dict.keys():
        g_sym_list.append(nlp_standard_full_dict["h"])
    g_sym = vertcat(*g_sym_list)
    # g_sym = ca.vertcat(nlp_standard_full_dict["g"], nlp_standard_full_dict["h"])

    nlp_standard_full = {"f":f_sym, "x":x_sym, "p":p_sym, "g":g_sym}
    
    return nlp_standard_full

if __name__ is '__main__':
    nlp, nlp_bounds, nlp_id = setup_NLP_example_1()

    nlp_diff = NLPDifferentiator(nlp, nlp_bounds)
    nlp_diff.transform_nlp(variant='full_standard')

    # Test 1
    p_num = np.array((0,0))

    if True:
        S0 = nlpsol('S', 'ipopt', nlp)
        r0 = S0(x0=0, p=p_num, **nlp_bounds)

        nlp_reconstruct = reconstruct_nlp(nlp_diff.nlp_standard_full_dict)
        S1 = nlpsol('S', 'ipopt', nlp_reconstruct)
        r1 = S1(x0=0, p=p_num, **nlp_diff.nlp_standard_full_bounds)

        # Print solution
        print(f'Solution r0x = {r0["x"]}, r1x = {r1["x"]}')

    # Test get Lagrangian
    nlp_diff.get_Lagrangian_sym()

    nlp_diff.prepare_hessian()

    H = nlp_diff.get_hessian(**r0)


    







    



  