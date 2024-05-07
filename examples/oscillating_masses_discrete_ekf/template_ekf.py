import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_ekf(model):
    """
    --------------------------------------------------------------------------
    template_ekf: tuning parameters
    --------------------------------------------------------------------------
    """
    

    # setting up model variances with a generic value
    q = 1e-3 * np.ones(model.x.shape)
    r = 1e-2 * np.ones(model.y.shape)
    Q = np.diag(q.flatten())
    R = np.diag(r.flatten())

    # inti
    ekf = do_mpc.estimator.EKF(model=model, Q=Q, R=R)

    # disc setup
    ekf.settings.n_horizon = 20

    # all parameters must be passed:
    p_template_mhe = ekf.get_p_template()
    def p_fun_ekf(t_now):
        p_template_mhe['Theta_1'] = 2.25e-4
        p_template_mhe['Theta_2'] = 2.25e-4
        p_template_mhe['Theta_3'] = 2.25e-4
        # And our previously set P_x:
        p_template_mhe['P_p'] = np.eye(1)
        return p_template_mhe
    ekf.set_p_fun(p_fun_ekf)

    # P_y is listed in the time-varying parameters and must be set.
    # This is more of a proof of concept (P_y is not actually changing over time).
    # We therefore do the following:
    tvp_template = ekf.get_tvp_template()
    tvp_template['_tvp', :, 'phi_2_set'] = np.diag(np.array([0]))
    tvp_template['_tvp', :, 'P_v'] = np.diag(np.array([1,1,1,20,20]))
    # Typically, the values would be reset at each call of tvp_fun.
    # Here we just return the fixed values:
    def tvp_fun_ekf(t_now):
        return tvp_template
    ekf.set_tvp_fun(tvp_fun_ekf)


    

    return ekf