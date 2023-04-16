import numpy as np
from casadi import *
from casadi.tools import *
import pdb
from ._nlphandler import NLPHandler

class NLPDifferentiator(NLPHandler):
    """
    Documentation for NLPDifferentiator.
 
    .. warning::

        This tool is currently not fully implemented and cannot be used.
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


    







    



  