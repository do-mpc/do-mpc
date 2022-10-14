import numpy as np
from casadi import *
from casadi.tools import *

class NLPHandler:
    """ 
    - Transform NLP in standard form.
    - Get Lagrangian of NLP
    - Get KKT conditions
    - Get metrics about NLP (number of constraints, decision variables etc.)
    
    """
    def __init__(self, nlp_dict, nlp_bounds):
        self.nlp_dict = nlp_dict
        self.nlp_bounds = nlp_bounds

        self.flags = {
            'transformed':    False,
            'get_Lagrangian': False,
            'get_KKT':        False,
        }

        #TODO: Können wir die bounds noch später ändern? Ja


    def transform_nlp(self):
        self.nlp_transformed_dict = None
        self.nlp_transformed_bounds = None
        
        # Function to map from original variables to transformed variables

        self.flags['transformed'] = True

    def get_Lagrangian_sym(self):
        self.Lagrangian_sym = None

        self.flags['get_Lagrangian'] = True

    def get_KKT_sym(self):

        self.KKT_full_sym = None
        self.KKT_equality_sym = None

        self.flags['get_KKT'] = True




class NLPDifferentiator(NLPHandler):
    """
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def prepare_hessian(self):
        # CasADi function to compute the Hessian of the Lagrangian with opt_x and opt_lam
        self.hessian_function = None

    def get_hessian(self, opt_x_num, opt_lam_num):
        # Compute Hessian with opt_x and opt_lam
        hessian = self.hessian_function(opt_x_num, opt_lam_num)

        return hessian

    
    def prepare_sensitivity(self):
        return
    
    def get_sensitivity(self, opt_x_num, opt_lam_num, opt_p_num):
        return

    def _get_A_matrix(self):
        return

    def _get_B_matrix(self):
        return

    

    

if __name__ is '__main__':
    pass
    





    



  